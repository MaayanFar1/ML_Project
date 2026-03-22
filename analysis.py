import os
import glob
import json
import warnings
import argparse

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from data.ring import RINGS_DICT

warnings.simplefilter(action='ignore', category=FutureWarning)

ATOMS_DICT = { "H", "C", "B", "N", "O", "S"}
# ---------------------- IO FUNCTIONS ------------------------
def load_all_csvs(csv_dir):
    """
    Load all molecule CSV files into a single pandas DataFrame.
    """
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in {csv_dir}")

    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        df["source_file"] = os.path.basename(file)
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    return full_df


# ---------------------- FILTERING ---------------------------

def filter_rings(
    df,
    max_nodes,
    ring_type=None,
    node_type=None,
    degree=None,
    orientation_only=False,
    num_rings=None,
):

    filtered = df.copy()

    if node_type is not None:
        filtered = filtered[filtered["type"] == node_type]

    if ring_type is not None:
        filtered = filtered[filtered["Ring_type"] == ring_type]

    if degree is not None:
        if isinstance(degree, list):
            filtered = filtered[filtered["degree"].isin(degree)]
        else:
            filtered = filtered[filtered["degree"] == degree]

    if num_rings is not None:
        # Count rings from the ORIGINAL df, not filtered
        real_rings_all = df[df["node_idx"] < max_nodes]

        ring_counts = real_rings_all.groupby("source_file")["node_idx"].count()

        valid_molecules = ring_counts[ring_counts == num_rings].index

        # Now filter the already-filtered dataframe
        filtered = filtered[filtered["source_file"].isin(valid_molecules)]


    if orientation_only:
        filtered = filtered[filtered["node_idx"] >= max_nodes]

    return filtered


# ---------------------- PLOTTING ---------------------------
def plot_iv_histogram(df, save_path, bins=50, label=None):
    """
    Plot smooth IV distribution using KDE with filled area.
    """

    iv_values = df["IV"].values
    mean = np.mean(iv_values)
    std = np.std(iv_values)

    if len(iv_values) < 2:
        print("Not enough data for KDE.")
        return

    kde = gaussian_kde(iv_values)

    x_limit = max(np.abs(iv_values.min()), np.abs(iv_values.max()))
    x = np.linspace((-1)*x_limit, x_limit, 500)
    y = kde(x) * len(iv_values)

    plt.figure(dpi=300)
    final_label = f"{label} ({mean:.3f} ± {std:.3f})" if label else f"μ={mean:.3f} ± {std:.3f}"
    plt.plot(x, y, color="blue", label=final_label)
    plt.fill_between(x, y, color="blue", alpha=0.5)
    plt.xlabel("IV value")
    plt.ylim(bottom=0)
    #plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_grouped_data(df, orientation_only, split_by_node_and_ring):
    """
    Returns an iterable of (label, sub_dataframe)
    depending on the selected grouping mode.
    """

    # atom and ring seperation
    if split_by_node_and_ring and orientation_only:
        grouped = df.groupby(["type", "Ring_type"])
        for (atom, ring), group in grouped:
            yield f"{atom}-{ring}", group, atom

    # atom / ring seperation
    else:
        if orientation_only:
            node_types = ATOMS_DICT
        else:
            node_types = RINGS_DICT

        for node_type in node_types:
            group = df[df["type"] == node_type]
            yield str(node_type), group, node_type


def plot_iv_histogram_by_ring_type(df, save_path, orientation_only, split_by_node_and_ring=False):

    if "type" not in df.columns:
        print("Column 'type' not found.")
        return

    df = df.dropna(subset=["IV", "type"])

    if len(df) == 0:
        print("Empty dataframe.")
        return

    # Common x-axis
    iv_values_all = df["IV"].values
    x_limit = max(np.abs(iv_values_all.min()), np.abs(iv_values_all.max()))
    x = np.linspace(-x_limit, x_limit, 500)

    plt.figure(dpi=300, figsize=(8, 6))

    plotted_any = False

    # 🔥 unified grouping
    for label_base, group, node_type in get_grouped_data(
        df, orientation_only, split_by_node_and_ring
    ):

        iv_values = group["IV"].values

        if len(iv_values) < 2:
            print(f"Skipping node_type={node_type}: not enough data for KDE.")
            continue

        mean = np.mean(iv_values)
        std = np.std(iv_values)

        try:
            kde = gaussian_kde(iv_values)
            y = kde(x) * len(iv_values)
        except Exception as e:
            print(f"Skipping ring_type={node_type}: KDE failed ({e})")
            continue

        label = f"{label_base} (n={len(iv_values)}, μ={mean:.3f} ± {std:.3f})"

        plt.plot(x, y, label=label)
        plotted_any = True
    # 🔥 Add aggregated node_type curves
    if split_by_node_and_ring and orientation_only:
        print("Adding aggregated node_type curves...")

        for node_type in ATOMS_DICT:
            group = df[df["type"] == node_type]
            iv_values = group["IV"].values

            if len(iv_values) < 2:
                print(f"Skipping node_type={node_type}: not enough data for KDE.")
                continue

            mean = np.mean(iv_values)
            std = np.std(iv_values)

            try:
                kde = gaussian_kde(iv_values)
                y = kde(x) * len(iv_values)
            except Exception as e:
                print(f"Skipping node_type={node_type}: KDE failed ({e})")
                continue

            # 🔥 distinguish visually
            label = f"{node_type} TOTAL (n={len(iv_values)}, μ={mean:.3f} ± {std:.3f})"

            plt.plot(
            x,
            y,
            linestyle="--", # 👈 key difference
            linewidth=2,
            label=label,
            )

            plotted_any = True

    if not plotted_any:
        print("No valid groups plotted.")
        plt.close()
        return

    plt.xlabel("IV value")
    # plt.ylabel("Density")
    plt.ylim(bottom=0)

    # Adjust legend size if many curves
    if split_by_node_and_ring and orientation_only:
        plt.legend(fontsize=6, ncol=2)
    else:
        plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
# ---------------------- ANALYSIS ----------------------------
def analyze(
    df,
    out_dir,
    max_nodes,
    ring_type=None,
    node_type=None,
    degree=None,
    orientation_only=False,
    num_rings=None,
    split_by_node_and_ring=False,
):
    df_filtered = filter_rings(
        df,
        max_nodes=max_nodes,
        ring_type=ring_type,
        node_type=node_type,
        degree=degree,
        orientation_only=orientation_only,
        num_rings=num_rings,
    )

    print(f"Rings after filtering: {len(df_filtered)}")

    if len(df_filtered) == 0:
        print("No rings left after filtering.")
        return

    os.makedirs(out_dir, exist_ok=True)

    title_parts = []
    if num_rings is not None:
        title_parts.append(f"num_rings={num_rings}")
    if orientation_only:
        title_parts.append("orientation_only")
        if split_by_node_and_ring:
            title_parts.append(f"sepereted")
    if node_type:
        title_parts.append(f"type={node_type}")
    if degree:
        title_parts.append(f"degree={degree}")
    if ring_type :
        title_parts.append(f"ring_type={ring_type}")



    label = ", ".join(title_parts) if title_parts else "All rings"

    safe_title = label.replace(" ", "_").replace(",", "").replace("=", "")
    save_path = os.path.join(out_dir, f"iv_histogram_{safe_title}.png")
    #save_path = os.path.join(args.out_dir, "iv_histogram.png") #make sure path is right

    print("Plotting histogram...")
    #plot_iv_histogram(df_filtered, save_path, bins=args.bins, label=label)
    plot_iv_histogram_by_ring_type(df_filtered, save_path, orientation_only, split_by_node_and_ring)
    print(f"Histogram saved to {save_path}")


def main(args):
    print("Loading CSV files...")
    df = load_all_csvs(args.csv_dir)
    print(f"Total rings loaded: {len(df)}")
    print("analyzing...")
    
    # ----------------------------------------
    # Define your experiment configurations here
    # ----------------------------------------
    configs = [
        dict(),
        dict(num_rings=9),
        dict(num_rings=9, degree=[1]),
        dict(num_rings=9, degree=[2]),
        dict(num_rings=9, degree=[3]),
        dict(orientation_only=True),
        dict(orientation_only=True, num_rings=9, degree=[1]),
        dict(orientation_only = True , num_rings = 9 , node_type = "N", split_by_node_and_ring = True),
        dict(orientation_only = True , num_rings = 9 , node_type = "B" , split_by_node_and_ring = True),
        # dict(orientation_only=True, num_rings=9, degree=[2]),
        # dict(orientation_only=True, num_rings=9, degree=[3]), Unrelavent null
    ]

    # ----------------------------------------
    # Run all experiments
    # ----------------------------------------
    for i, cfg in enumerate(configs):
        print(f"### Experiment {i+1}/{len(configs)} ###")

        analyze(
            df=df,
            out_dir=args.out_dir,
            max_nodes=args.max_nodes,
            ring_type=cfg.get("ring_type"),
            node_type=cfg.get("node_type"),
            degree=cfg.get("degree"),
            orientation_only=cfg.get("orientation_only", False),
            num_rings=cfg.get("num_rings"),
            split_by_node_and_ring=cfg.get("split_by_node_and_ring", False),
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_dir", type=str, default="/home/maayanfarkash/proj/prediction_summary/hetro/hetro/analysis")
    parser.add_argument("--out_dir", type=str, default="analysis_output")

    parser.add_argument("--max_nodes", type=int, default=15)

    #parser.add_argument("--ring_type", type=str, default=None)
    #parser.add_argument("--node_type", type=str, default=None)
    #parser.add_argument("--num_rings", type=int, default=None)
    #parser.add_argument("--degree", type=int, nargs="+", default=None)
    #parser.add_argument("--orientation_only", action="store_true")

    parser.add_argument("--bins", type=int, default=50)

    args = parser.parse_args()

    main(args)