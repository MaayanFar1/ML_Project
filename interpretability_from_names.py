import json
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.aromatic_dataloader import create_data_loaders
from gradram import plot_mol_gradram_from_tensors, grad_ram
from train_clean_predictor import get_cond_predictor_model
from utils.utils_edm import normalize

import warnings

from args import Args

warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import pandas as pd


def try_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def interpretation(model, dataloader, args, target_idx):
    model.eval()
    # load pyrenes
    # pyrenes_df = pd.read_csv(
    #     "/home/maayanfarkash/proj/PBHs-design/compas-3D_pyrenes.csv"
    # )
    # # adjust column name if needed
    # names = pyrenes_df["molecule"].tolist()
    names = ["C2M465519"]

    df = dataloader.dataset.df

    # Optional: keep only molecules that exist in your dataset
    names = [n for n in names if n in set(df.molecule.values)]

    print(f"Running interpretation on {len(names)} pyrene molecules")

    # dir_name = f'{args.exp_dir}/interp_Pyrenes_{args.target_features.split(",")[target_idx]}'
    dir_name = f'{args.exp_dir}/hetro_try'
    try_mkdir(dir_name)

    for i, name in enumerate(names):
        # pdf_filename = f'{dir_name}/{args.target_features.split(",")[target_idx]}-{name}.pdf'
        pdf_filename= f'{dir_name}/hetro_try-{name}.pdf'
        if os.path.isfile(pdf_filename):
            print(i, name, "exists -> skip")
            continue

        df_row = df[df.molecule == name].iloc[0]
        mol, edges, atom_connectivity, _ = dataloader.dataset.get_mol(df_row)
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
        pred[0, target_idx].backward()

        final_conv_acts = model.final_conv_acts
        final_conv_grads = model.final_conv_grads
        grad_ram_weights = grad_ram(final_conv_acts, final_conv_grads, normalize=False)

        fig = plot_mol_gradram_from_tensors(
            x_full, node_mask, mol, edges, grad_ram_weights,
            value=y_cpu[0, target_idx].item(),
            target_features=args.target_features.split(",")[target_idx],
            max_nodes=args.max_nodes,
        )

        fig.savefig(pdf_filename, bbox_inches="tight")
        plt.close(fig)


def main(args):
    train_loader, val_loader, test_loader = create_data_loaders(args)

    if not args.restore:
        print("FLAGS.restore must be set")

    model = get_cond_predictor_model(args, val_loader.dataset)

    print("Begin evaluation")
    interpretation(model, train_loader, args, target_idx=0)


if __name__ == '__main__':
    args = Args().parse_args()

    args.exp_dir = f"{args.save_dir}/{args.name}"
    with open('/home/maayanfarkash/proj/prediction_summary/hetro/args_clean.txt', "r") as f:
        args.__dict__ = json.load(f)

    args.restore = True
    args.transform = False
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    main(args)
