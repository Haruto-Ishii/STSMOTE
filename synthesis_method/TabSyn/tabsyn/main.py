import os
import torch
import numpy as np
import random

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time

from tqdm import tqdm
from synthesis_method.TabSyn.tabsyn.model import MLPDiffusion, Model
from synthesis_method.TabSyn.tabsyn.latent_utils import get_input_train
from synthesis_method.TabSyn.tabsyn.vae.main import main as vae_main
from synthesis_method.TabSyn.tabsyn.diffusion_utils import sample
from synthesis_method.TabSyn.tabsyn.latent_utils import split_num_cat_target

warnings.filterwarnings('ignore')


def main(X_minority, y_minority, X_majority, y_majority, num_to_generate, info, random_state): 
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    vae_decoder, train_z, num_inverse, cat_inverse = vae_main(X_minority, y_minority, X_majority, y_majority, num_to_generate, info, random_state)
    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    train_z = train_z.view(B, in_dim)
    info['pre_decoder'] = vae_decoder
    info['token_dim'] = token_dim

    in_dim = train_z.shape[1] 

    mean, std = train_z.mean(0), train_z.std(0)

    train_z = (train_z - mean) / 2
    train_data = train_z.cpu()

    batch_size = min(len(y_minority), 20)
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0
    )

    num_epochs = 10000 + 1

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    print(denoise_fn)

    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)

    model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20)

    model.train()

    best_loss = float('inf')
    patience = 0
    for epoch in range(num_epochs):
        
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            inputs = batch.float().to(device)
            loss = model(inputs)
        
            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

        curr_loss = batch_loss/len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            best_model = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience == 500:
                print('Early stopping')
                break

    
    model.load_state_dict(best_model)
    
    sample_dim = in_dim
    x_next = sample(model.denoise_fn_D, num_to_generate, sample_dim)
    x_next = x_next * 2 + mean.to(device)
    
    syn_data = x_next
    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse)
    
    X_synthetic = np.hstack([syn_num, syn_cat])
    
    return X_synthetic, syn_target.ravel()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TabSyn')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'