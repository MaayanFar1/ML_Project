import json
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.aromatic_dataloader import create_data_loaders
from gradram import plot_mol_gradram, grad_ram
from train_clean_predictor import get_cond_predictor_model
from utils.utils_edm import normalize

import warnings

from args import Args

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import torch


def try_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def to_np(x):
    return x.cpu().detach().numpy()

def interpretation(model, dataloader, args):
    model.eval()
    samples = range(len(dataloader.dataset.df_all))
    samples=np.random.permutation(samples)

    for i in samples:
        df_row = dataloader.dataset.df_all.iloc[i]
        mol, edges, name = dataloader.dataset.get_mol(df_row)
        pdf_filename = f'{args.exp_dir}/interp-{args.target_features}/{args.target_features}-{name}.pdf'
        if os.path.isfile(pdf_filename):
            print(i)
            continue
        else:
            print(i, name)
            
            # we changed to the egnn model 
            x_full, node_mask, edge_mask, node_features_full, y, adj_full = dataloader.dataset.get_all(df_row)
            y = y.to(args.device)
            x_full = x_full.to(args.device)
            node_features_full = node_features_full.to(args.device)
            node_mask = node_mask.to(args.device)
            edge_mask = edge_mask.to(args.device)
            adj_full = adj_full.to(args.device)

            x_norm, h_norm, _ = normalize(
                x_full,
                {"categorical": node_features_full, "integer": torch.zeros(0, device=x.device)},
                node_mask,
            )
            xh = torch.cat([x_norm, h_norm["categorical"]], dim=-1)  # Build [x_norm | h_norm] input, size [bs, n_nodes, d+in_nf]
            bs, n_nodes, _ = x_full.shape
            edge_mask_flat = edge_mask.view(bs, n_nodes * n_nodes)   # [bs, n_nodes^2]
            pred = model(xh, node_mask , edge_mask_flat, adj_full )

            y = y.cpu() * dataloader.dataset.std + dataloader.dataset.mean
            pred = pred.cpu() * dataloader.dataset.std + dataloader.dataset.mean
            pred.backward()

            final_conv_acts = model.final_conv_acts.view(-1, 96)
            final_conv_grads = model.final_conv_grads.view(-1, 96)
            grad_ram_weights = grad_ram(final_conv_acts, final_conv_grads, False)
            fig = plot_mol_gradram(g, mol, edges, grad_ram_weights, y.item(), args.target_features)
            fig.savefig(pdf_filename, bbox_inches='tight')
            # fig.show()
            plt.close(fig)

def main(args):
    # Prepare data
    train_loader, val_loader, test_loader = create_data_loaders(args)

    # Choose model
    if not args.restore:
        print("FLAGS.restore must be set")
    model = get_cond_predictor_model(args, val_loader.dataset)

    # Run training
    print('Begin evaluation')
    interpretation(model, train_loader, args)

if __name__ == '__main__':
    args = Args().parse_args()

    # torch.manual_seed(0)
    # np.random.seed(0)

    args.name='Erel'
    # args.name = 'SE3-knots-GAP_eV'
    print(args.name)
    args.exp_dir = f'{args.save_dir}/{args.name}'
    with open(args.exp_dir + '/args.txt', "r") as f:
        args.__dict__ = json.load(f)
    args.restore = True
    args.transform = False
    # Automatically choose GPU if available
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("\n\nArgs:", args)

    main(args)
