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

warnings.simplefilter(action='ignore', category=FutureWarning)


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
    degree=None,
    orientation_only=False,
):
    """
    Filter rings based on:
        - ring_type (exact match)
        - degree (int or list of ints)
        - orientation_only (node_idx >= max_nodes)
    """

    filtered = df.copy()

    if ring_type is not None:
        filtered = filtered[filtered["type"] == ring_type]

    if degree is not None:
        if isinstance(degree, list):
            filtered = filtered[filtered["degree"].isin(degree)]
        else:
            filtered = filtered[filtered["degree"] == degree]

    if orientation_only:
        filtered = filtered[filtered["node_idx"] >= max_nodes]

    return filtered


# ---------------------- HISTOGRAM ---------------------------
def plot_iv_histogram(df, save_path, bins=50, title=None):
    """
    Plot histogram of IV values.
    """
    iv_values = df["IV"].values

    plt.figure()
    plt.hist(iv_values, bins=bins)
    plt.xlabel("IV value")
    plt.ylabel("Count")

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ---------------------- ANALYSIS ----------------------------
def analyze(args, df):

    df_filtered = filter_rings(
        df,
        max_nodes=args.max_nodes,
        ring_type=args.ring_type,
        degree=args.degree,
        orientation_only=args.orientation_only,
    )

    print(f"Rings after filtering: {len(df_filtered)}")

    if len(df_filtered) == 0:
        print("No rings left after filtering.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    title_parts = []
    if args.ring_type:
        title_parts.append(f"type={args.ring_type}")
    if args.degree:
        title_parts.append(f"degree={args.degree}")
    if args.orientation_only:
        title_parts.append("orientation_only")

    title = ", ".join(title_parts) if title_parts else "All rings"

    save_path = os.path.join(args.out_dir, "iv_histogram.png") #make sure path is right

    print("Plotting histogram...")
    plot_iv_histogram(df_filtered, save_path, bins=args.bins, title=title)

    print(f"Histogram saved to {save_path}")


def main(args):
    print("Loading CSV files...")
    df = load_all_csvs(args.csv_dir)
    print(f"Total rings loaded: {len(df)}")
    print("analyzing...")
    analyze(args, df)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="analysis_output")

    parser.add_argument("--max_nodes", type=int, default=15)

    parser.add_argument("--ring_type", type=str, default=None)
    parser.add_argument("--degree", type=int, nargs="+", default=None)
    parser.add_argument("--orientation_only", action="store_true")

    parser.add_argument("--bins", type=int, default=50)

    args = parser.parse_args()

    main(args)