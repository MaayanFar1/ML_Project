import os
from pathlib import Path
import torch
import networkx as nx
from tqdm import tqdm

from prediction_args import PredictionArgs
from data.aromatic_dataloader import AromaticDataset, get_splits, get_paths, DTYPE
from utils.ring_graph import get_rings, get_rings_adj
from torch.nn.functional import one_hot
#

def preprocess_one(dataset, df_row):
    name = df_row["molecule"]
    preprocessed_dir = dataset.xyz_root + "_rings_preprocessed"
    os.makedirs(preprocessed_dir, exist_ok=True)

    preprocessed_path = os.path.join(preprocessed_dir, name + ".pt")
    if Path(preprocessed_path).is_file():
        return "exists", name

    mol, edges, atom_connectivity, _ , rd_mol = dataset.get_mol(df_row, skip_hydrogen=False)
    mol_graph = nx.Graph(edges)

    knots, knots_with_orientation = get_rings(rd_mol , mol , mol.atoms, mol_graph)
    adj = get_rings_adj(knots)
    x = torch.tensor([k.get_coord() for k in knots], dtype=DTYPE)

    knot_type = torch.tensor(
        [dataset.knots_list.index(k.cycle_type) for k in knots]
    ).unsqueeze(1)

    node_features = one_hot(
        knot_type, num_classes=len(dataset.knots_list)
    ).squeeze(1).float()

    orientation = [k.orientation for k in knots]

    torch.save(
        [x, adj, node_features, orientation, knots_with_orientation],
        preprocessed_path
    )
    return "saved", name


def main():
    parser = PredictionArgs()
    args = parser.parse_args()

    # build dataset splits exactly like training
    args.df_train, args.df_val, args.df_test, args.df_all = get_splits(args)

    # choose which split(s) to preprocess
    datasets = {
        "train": AromaticDataset(args=args, task="train"),
        "val": AromaticDataset(args=args, task="val"),
        "test": AromaticDataset(args=args, task="test"),
    }

    for split_name, dataset in datasets.items():
        print(f"\nPreprocessing split: {split_name}")
        saved_count = 0
        exists_count = 0
        error_count = 0

        for i in tqdm(range(len(dataset)), desc=f"{split_name}"):
            index = dataset.examples[i]
            df_row = dataset.df.loc[index]

            try:
                status, name = preprocess_one(dataset, df_row)
                if status == "saved":
                    saved_count += 1
                else:
                    exists_count += 1
            except Exception as e:
                error_count += 1
                print(f"[ERROR] split={split_name}, idx={i}, molecule={df_row['molecule']}, error={e}")

        print(
            f"{split_name}: saved={saved_count}, exists={exists_count}, errors={error_count}"
        )


if __name__ == "__main__":
    main()