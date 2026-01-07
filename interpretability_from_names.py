import json
import os
import matplotlib.pyplot as plt

from data.aromatic_dataloader import create_data_loaders
from gradram import plot_mol_gradram, grad_ram
from train_clean_predictor import get_cond_predictor_model

import warnings

from args import Args

warnings.simplefilter(action='ignore', category=FutureWarning)

import torch

def try_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def to_np(x):
    return x.cpu().detach().numpy()

def interpretation(model, dataloader, args):
    model.eval()
    names = [
        # # linear
        # 'hc_c34h20_0pent_442', 'hc_c38h22_0pent_1677', 'hc_c34h20_0pent_434', 'hc_c34h20_0pent_425',
        # # angular
        # 'hc_c34h20_0pent_62', 'hc_c34h20_0pent_29', 'hc_c34h20_0pent_33', 'hc_c34h20_0pent_41',
        # 'hc_c42h24_0pent_950', 'hc_c38h22_0pent_1053', 'hc_c38h22_0pent_1046',
        # Branched
        'hc_c38h22_0pent_159', 'hc_c34h20_0pent_193', 'hc_c30h18_0pent_56', 'hc_c30h18_0pent_57',
        # 2 sample
        # 'hc_c38h22_0pent_1009', 'hc_c34h20_0pent_274', 'hc_c34h20_0pent_177',
        # 'hc_c38h22_0pent_1295', 'hc_c34h20_0pent_337', 'hc_c34h20_0pent_183',
        # # old slide
        # 'hc_c34h20_0pent_123', 'hc_c34h20_0pent_121', 'hc_c42h24_0pent_103',
        # 'hc_c42h24_0pent_1015', 'hc_c38h22_0pent_105', 'hc_c38h22_0pent_100',
        # # cove
        # 'hc_c38h22_0pent_170', 'hc_c38h22_0pent_171', 'hc_c38h22_0pent_576',
        # # fjord
        # 'hc_c34h20_0pent_14', 'hc_c38h22_0pent_55', 'hc_c38h22_0pent_575',
        # # helix
        # 'hc_c42h24_0pent_110',  'hc_c38h22_0pent_360', 'hc_c38h22_0pent_408'
    ]

    df = dataloader.dataset.df_all
    dir_name = f'{args.exp_dir}/interp-{args.target_features}'
    try_mkdir(dir_name)

    for i, name in enumerate(names):
        pdf_filename = f'{dir_name}/{args.target_features}-{name}.pdf'
        if os.path.isfile(pdf_filename):
            print(i)
            continue
        else:
            df_row = df[df.molecule == name].iloc[0]
            mol, edges, name = dataloader.dataset.get_mol(df_row) ##TODO: add edges to Gaudi.
            title = f'{args.target_features}-{name}'
            print(i, name)
            ##TODO: we changed to the egnn model 
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

            # 2) Build [x_norm | h_norm] input
            xh = torch.cat([x_norm, h_norm["categorical"]], dim=-1)  # [bs, n_nodes, d+in_nf]
            
            bs, n_nodes, _ = x_full.shape
            edge_mask_flat = edge_mask.view(bs, n_nodes * n_nodes)   # [bs, n_nodes^2]

            pred = model(xh, node_mask , edge_mask_flat, adj_full )

            y = y.cpu() * dataloader.dataset.std + dataloader.dataset.mean ##TODO: check how to normlize the dataset from Gaudi
            pred = pred.cpu() * dataloader.dataset.std + dataloader.dataset.mean ##TODO 
            pred.backward()

            final_conv_acts = model.final_conv_acts.view(-1, 96) ## TODO: check how to take the final conv layer from GaUdi. 
            final_conv_grads = model.final_conv_grads.view(-1, 96)
            grad_ram_weights = grad_ram(final_conv_acts, final_conv_grads, False)
            fig = plot_mol_gradram(g, mol, edges, grad_ram_weights, y.item(), args.target_features)
            fig.savefig(pdf_filename, bbox_inches='tight')
            fig.show()
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

    args.name='Erel'
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
