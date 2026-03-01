import json
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.aromatic_dataloader import create_data_loaders
from gradram import plot_mol_gradram_from_tensors, grad_ram
from train_clean_predictor import get_cond_predictor_model
from utils.utils_edm import normalize
from data.ring import RINGS_DICT

import warnings
import numpy as np
import pandas as pd

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
    names = ["C2M524995"]

    df = dataloader.dataset.df

    # Optional: keep only molecules that exist in your dataset
    names = [n for n in names if n in set(df.molecule.values)]

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

        save_molecule_node_csv(name , x_full , node_features_full , grad_ram_weights, dir_name , node_mask , args.max_nodes)

        fig.savefig(pdf_filename, bbox_inches="tight")
        plt.close(fig)


def ensure_np(a):
    """Torch tensor -> numpy; numpy stays numpy."""
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)

def node_type_from_onehot(node_features_row):
    """
    node_features_row: shape (C,) one-hot or soft one-hot.
    Returns (type_index, type_name)
    """
    RING_NAMES = list(RINGS_DICT.keys())
    type_idx = int(np.argmax(node_features_row))
    type_name = RING_NAMES[type_idx]
    return type_idx, type_name

def save_molecule_node_csv(
    name: str,
    x_full,
    node_features_full,
    gradramweights,
    out_dir: str,
    node_mask,
    max_nodes
  ):  
    """
    Saves: out_dir/<name>.csv
    Columns: node_idx, x, y, z, type_idx, type_name, gradram_weight
    Keeps rows where x_full row is not all zeros (optionally with eps tolerance).
    """
    os.makedirs(out_dir, exist_ok=True)

    x = ensure_np(x_full)
    nf = ensure_np(node_features_full)
    w = ensure_np(gradramweights)


    # Handle common shapes:
    if x.ndim == 3:
        x = x[0]
    if nf.ndim == 3:
        nf = nf[0]
    if w.ndim == 2 and w.shape[0] == 1:
        w = w[0]
    
    RING_NAMES = list(RINGS_DICT.keys())
    # print("len(RING_NAMES):", len(RING_NAMES))
    # print("max type_idx in batch:", int(np.argmax(nf, axis=1).max()))

    w = w.reshape(-1)
    nm = ensure_np(node_mask)
    if nm.ndim == 3:  # (1,N,1)
        nm = nm[0]
    if nm.ndim == 2 and nm.shape[1] == 1:
        nm = nm[:, 0]
    keep = nm.astype(bool)

    idxs = np.where(keep)[0]
    idxs = idxs[idxs < max_nodes]

    rows = []
    for i in idxs:
        type_idx, type_name = node_type_from_onehot(nf[i])
        rows.append({
            "node_idx": int(i),
            "x": float(x[i, 0]),
            "y": float(x[i, 1]),
            "z": float(x[i, 2]),
            "type": type_name,
            "IV": float(w[i]),
        })

    idxs = np.where(keep)[0]
    idxs = idxs[idxs >= max_nodes]

    for i in idxs:
        type_idx, type_name = node_type_from_onehot(nf[i-max_nodes])
        last_char = RINGS_DICT[type_name][-1]
        rows.append({
            "node_idx": i,
            "x": float(x[i, 0]),
            "y": float(x[i, 1]),
            "z": float(x[i, 2]),
            "type": last_char,
            "IV": float(w[i]),
            "Ring_indx": i-max_nodes
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

def save_all_molecules_csv(
    names,
    x_full_list,
    node_features_full_list,
    gradramweights_list,
    out_dir,
    knots_list=None,
):
    """
    If you have per-molecule arrays already in lists aligned with `names`.
    """
    paths = []
    for name, x_full, nf_full, w in zip(names, x_full_list, node_features_full_list, gradramweights_list):
        paths.append(save_molecule_node_csv(
            name=name,
            x_full=x_full,
            node_features_full=nf_full,
            gradramweights=w,
            out_dir=out_dir,
            knots_list=knots_list,
        ))
    return paths


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
