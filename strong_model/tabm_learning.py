import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from typing import Any, Literal, NamedTuple, Optional, List, Dict
import tabm
import rtdl_num_embeddings
import math
from copy import deepcopy
import scipy.special
import sklearn.metrics
from torch import Tensor
import random

def main(X_train_full, y_train_full, X_test, y_test, num_numerical_features, random_state):
    
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_idx, val_idx = train_test_split(
        np.arange(len(y_train_full)), train_size=0.8, shuffle=True, stratify=y_train_full, random_state=random_state
    )
    
    data_numpy = {
        'train': {'x': X_train_full[train_idx], 'y': y_train_full[train_idx]},
        'val': {'x': X_train_full[val_idx], 'y': y_train_full[val_idx]},
        'test': {'x': X_test, 'y': y_test}
    }
    
    TaskType = Literal['regression', 'binclass', 'multiclass']
    task_type = 'multiclass'
    
    unique_labels = np.unique(y_train_full)
    n_classes = len(unique_labels)
    
    assert set(unique_labels) == set(range(n_classes)), (
        f"Labels must be in the range [0, {n_classes - 1}]"
    )
    
    for part in data_numpy:
        data_numpy[part]['y'] = data_numpy[part]['y'].astype(np.int64)
    
    task_is_regression = False

    n_num_features = num_numerical_features
    n_cat_features = X_train_full.shape[1] - n_num_features
    
    data_numpy_processed = {'train': {}, 'val': {}, 'test': {}}
    
    for part in data_numpy:
        data_numpy_processed[part]['x_num'] = data_numpy[part]['x'][:, :n_num_features].astype(np.float32)
        if n_cat_features > 0:
            data_numpy_processed[part]['x_cat'] = data_numpy[part]['x'][:, n_num_features:].astype(np.int64)
        data_numpy_processed[part]['y'] = data_numpy[part]['y']
        
    data_numpy = data_numpy_processed
    x_num_train_numpy = data_numpy['train']['x_num']
    
    preprocessing = QuantileTransformer(
        n_quantiles=max(min(len(x_num_train_numpy) // 30, 1000), 10),
        output_distribution='normal',
        subsample=10**9,
        random_state=random_state
    ).fit(x_num_train_numpy)
    
    for part in data_numpy:
        data_numpy[part]['x_num'] = preprocessing.transform(data_numpy[part]['x_num'])
        
    Y_train_numpy = data_numpy['train']['y'].copy()
    regression_label_stats = None

    if n_cat_features > 0:
        
        x_cat_train_full_numpy = X_train_full[:, n_num_features:].astype(np.int64)
        x_cat_test_numpy = X_test[:, n_num_features:].astype(np.int64)
        
        x_cat_combined_numpy = np.vstack([x_cat_train_full_numpy, x_cat_test_numpy])

        cat_cardinalities = [
            int(np.max(x_cat_combined_numpy[:, i])) + 1 for i in range(n_cat_features)
        ]
    else:
        cat_cardinalities = []
    
    num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
        rtdl_num_embeddings.compute_bins(torch.as_tensor(data_numpy['train']['x_num']), n_bins=48),
        d_embedding=16,
        activation=False,
        version='B',
    )
    
    model = tabm.TabM.make(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        d_out=n_classes,
        num_embeddings=num_embeddings,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
    gradient_clipping_norm: Optional[float] = 1.0
    
    base_loss_fn = F.cross_entropy
    
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else None
    )
    amp_enabled = False and amp_dtype is not None
    grad_scaler = torch.amp.GradScaler("cuda") if amp_dtype is torch.float16 else None

    compile_model = False
    if compile_model:
        model = torch.compile(model)
        evaluation_mode = torch.no_grad
    else:
        evaluation_mode = torch.inference_mode

    data = {
        part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
        for part in data_numpy
    }
    Y_train = torch.as_tensor(Y_train_numpy, device=device)

    share_training_batches = True

    @torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
    def apply_model(part: str, idx: Tensor) -> Tensor:
        return (
            model(
                data[part]['x_num'][idx],
                data[part]['x_cat'][idx] if 'x_cat' in data[part] else None,
            )
            .float()
        )

    def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
        y_pred = y_pred.flatten(0, 1)

        if share_training_batches:
            y_true = y_true.repeat_interleave(model.backbone.k)
        else:
            y_true = y_true.flatten(0, 1)

        return base_loss_fn(y_pred, y_true)

    @evaluation_mode()
    def evaluate(part: str) -> float:
        model.eval()
        eval_batch_size = 8096
        y_pred: np.ndarray = (
            torch.cat(
                [
                    apply_model(part, idx)
                    for idx in torch.arange(len(data[part]['y']), device=device).split(
                        eval_batch_size
                    )
                ]
            )
            .cpu()
            .numpy()
        )
        
        y_pred = scipy.special.softmax(y_pred, axis=-1)
        y_pred = y_pred.mean(1)
        
        y_true = data[part]['y'].cpu().numpy()
        score = sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1))
        return float(score)

    n_epochs = 1000
    train_size = len(train_idx)
    batch_size = 256
    epoch_size = math.ceil(train_size / batch_size)
    patience = 16

    epoch = -1
    metrics = {'val': -math.inf, 'test': -math.inf}

    def make_checkpoint() -> dict[str, Any]:
        return deepcopy(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'metrics': metrics,
            }
        )

    best_checkpoint = make_checkpoint()
    remaining_patience = patience

    for epoch in range(n_epochs):
        if share_training_batches:
            indices = torch.randperm(train_size, device='cpu').to(device)
            batches = indices.split(batch_size)
        else:
            rand_vals = torch.rand((train_size, model.backbone.k), device='cpu')
            indices = rand_vals.argsort(dim=0).to(device)
            batches = indices.split(batch_size, dim=0)
        # batches = (
        #     torch.randperm(train_size, device=device).split(batch_size)
        #     if share_training_batches
        #     else (
        #         torch.rand((train_size, model.backbone.k), device=device)
        #         .argsort(dim=0)
        #         .split(batch_size, dim=0)
        #     )
        # )
        
        for batch_idx in batches:
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(apply_model('train', batch_idx), Y_train[batch_idx])
            
            if gradient_clipping_norm is not None:
                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), gradient_clipping_norm
                )
            
            if grad_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

        metrics = {part: evaluate(part) for part in ['val', 'test']}
        val_score_improved = metrics['val'] > best_checkpoint['metrics']['val']

        print(
            f'{"*" if val_score_improved else " "}'
            f' [epoch] {epoch:<3}'
            f' [val] {metrics["val"]:.4f}'
            f' [test] {metrics["test"]:.4f}'
        )

        if val_score_improved:
            best_checkpoint = make_checkpoint()
            remaining_patience = patience
        else:
            remaining_patience -= 1

        if remaining_patience < 0:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_checkpoint['model'])

    print('\n[Summary]')
    print(f'best epoch:  {best_checkpoint["epoch"]}')
    print(f'val score:   {best_checkpoint["metrics"]["val"]:.4f}')
    print(f'test score:  {best_checkpoint["metrics"]["test"]:.4f}')

    @evaluation_mode()
    def get_predictions(part: str) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        eval_batch_size = 8096
        
        y_pred_raw: np.ndarray = (
            torch.cat(
                [
                    apply_model(part, idx)
                    for idx in torch.arange(len(data[part]['y']), device=device).split(
                        eval_batch_size
                    )
                ]
            )
            .cpu()
            .numpy()
        )
        
        y_pred_prob_all_k = scipy.special.softmax(y_pred_raw, axis=-1)
        final_probabilities = y_pred_prob_all_k.mean(axis=1)
        predicted_classes = final_probabilities.argmax(axis=1)
        
        return predicted_classes, final_probabilities

    predicted_classes_test, probabilities_test = get_predictions('test')
    
    return predicted_classes_test, probabilities_test