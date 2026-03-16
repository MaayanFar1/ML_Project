import json
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.aromatic_dataloader import create_data_loaders
from gradram import plot_mol_gradram_from_tensors, grad_ram
from train_clean_predictor import get_cond_predictor_model
from utils.utils_edm import normalize
from utils.helpers import save_molecule_node_csv

import warnings

from args import Args

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import torch


# def to_np(x):
#     return x.cpu().detach().numpy()


def interpretation(model, dataloader, args, target_idx):
    model.eval()
    samples = range(len(dataloader.dataset.df))
    samples=np.random.permutation(samples)

    # out_dir = f'/home/maayanfarkash/proj/prediction_summary/hetro/interp_{args.target_features.split(",")[target_idx]}'
    dir_name = f'{args.exp_dir}/hetro'
    os.makedirs(dir_name, exist_ok=True)
    os.makedirs(f"{dir_name}/figures", exist_ok=True)
    os.makedirs(f"{dir_name}/analysis", exist_ok=True)

    for i in samples[:500]:
        df_row = dataloader.dataset.df.iloc[i]
        mol, edges, atom_connectivity, name = dataloader.dataset.get_mol(df_row)

        fig_path = f'{dir_name}/figures/{args.target_features.split(",")[target_idx]}-{name}.pdf'
        csv_path = f'{dir_name}/analysis/{args.target_features.split(",")[target_idx]}-{name}.csv'

        if os.path.isfile(fig_path):
            print(i, "fig exists -> skip")
            continue

        if os.path.isfile(csv_path):
            print(i, "analysis data exists -> skip")
            continue

        print(i, name)

        x_full, node_mask, edge_mask, node_features_full, y, adj_full = dataloader.dataset.get_all(df_row)

        y = y.to(args.device).unsqueeze(0)
        x_full = x_full.to(args.device).unsqueeze(0)
        node_features_full = node_features_full.to(args.device).unsqueeze(0)
        node_mask = node_mask.to(args.device).unsqueeze(0).unsqueeze(2)
        adj_full = adj_full.to(args.device).unsqueeze(0)

        x_norm, h_norm, _ = normalize(
            x_full,
            {"categorical": node_features_full, "integer": torch.zeros(0, device=x_full.device)},
            node_mask,
        )
        xh = torch.cat([x_norm, h_norm["categorical"]], dim=-1)

        bs, n_nodes, _ = x_full.shape
        edge_mask_flat = edge_mask.to(args.device).view(1, n_nodes * n_nodes)

        model.zero_grad(set_to_none=True)
        pred = model(xh, node_mask, edge_mask_flat, adj_full)

        y_cpu = y.cpu() * dataloader.dataset.std + dataloader.dataset.mean
        pred_cpu = pred.detach().cpu() * dataloader.dataset.std + dataloader.dataset.mean
        # backprop a scalar
        pred[0, target_idx].backward()

        final_conv_acts = model.final_conv_acts
        final_conv_grads = model.final_conv_grads
        grad_ram_weights = grad_ram(final_conv_acts, final_conv_grads, normalize=False)

        # -------- Save CSV --------
        save_molecule_node_csv(csv_path, x_full, node_features_full, grad_ram_weights, node_mask, adj_full, args.max_nodes)

        # -------- Save Figure --------
        fig = plot_mol_gradram_from_tensors(
            x_full, node_mask, mol, edges, grad_ram_weights,
            value=y_cpu[0, target_idx].item(),
            target_features=args.target_features.split(",")[target_idx],
            max_nodes=args.max_nodes
        )

        fig.savefig(fig_path, bbox_inches="tight")
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
    interpretation(model, train_loader, args, target_idx=2)

if __name__ == '__main__':
    args = Args().parse_args()

    args.exp_dir = f'{args.save_dir}/{args.name}'
    with open('/home/maayanfarkash/proj/prediction_summary/hetro/args_clean.txt', "r") as f:
        args.__dict__ = json.load(f)
    args.restore = True
    args.transform = False
    # Automatically choose GPU if available
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    main(args)
