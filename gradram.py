import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# import pickle
# import torch.nn.functional as F
# from torch import nn, Tensor
# from adjustText import adjust_text
# from data.mol import Mol


def grad_ram(final_conv_acts, final_conv_grads, normalize=True):
    node_heat_map = []
    alphas = torch.mean(final_conv_grads, axis=0) # mean gradient for each feature (512x1)
    for n in range(final_conv_acts.shape[0]): # nth node
        node_heat = (alphas @ final_conv_acts[n]).item()
        node_heat_map.append(node_heat)
    node_heat_map = np.array(node_heat_map)
    if normalize:
        node_heat_map = node_heat_map / np.abs(node_heat_map).max()
    return node_heat_map


def align_manual(x, rotation_angle):
    rotation_angle = np.deg2rad(rotation_angle)
    # print(np.rad2deg(rotation_angle))
    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    Vt = np.array([[c, -s], [s, c]])
    return Vt

def align_to_x_plane(x):
    """
    Rotate the molecule into x axis.
    """
    x = x[:,:2]
    xm = x.mean(0)
    xc = x - xm
    _, _, Vt = np.linalg.svd(xc)
    return Vt

def flip(x, x_atoms, v, h):
    if v:
        x[:,1] *= -1
        x_atoms[:,1] *= -1
    if h:
        x[:,0] *= -1
        x_atoms[:,0] *= -1
    return x, x_atoms

def moldraw(ax,xr, _molrepr, _edges, plot_h=False):
    """
    moldraw(ax, _molrepr: Mol Object, _edges: list)

    Function that draws the molecule.

    in:
    _molrepr: Mol Object containing XYZ data.
    _edges: list of tuples containing the indices of 2 atoms bonding.

    """

    atom_colors = {'H':'silver','N':'blue','O':'red','S':'goldenrod','B':'green'}

    # plot molecule
    for edge in _edges:
        bond = []
        Hbond = False
        for atom_idx in edge:
            for atom in _molrepr.atoms:
                if atom_idx == atom.index:
                    bond.append(atom)
                    if atom.element != 'C':
                        if atom.element == 'BH':
                            atom.element = 'B'
                        elif atom.element == 'H':
                            Hbond = True
                        if Hbond and not plot_h:
                            continue
                        ax.text(xr[atom_idx, 0], xr[atom_idx, 1], atom.element, ha='center', va='center', color=atom_colors[atom.element],
                                zorder=2, bbox=dict(facecolor='white', edgecolor='none', boxstyle='circle, pad=0.1'))
        x = [xr[atom.index, 0] for atom in bond]
        y = [xr[atom.index, 1] for atom in bond]
        if Hbond:
            if plot_h:
                ax.plot(x, y, c=atom_colors['H'], linestyle='-', zorder=0)
        else:
            ax.plot(x, y, c='black', linestyle='-', zorder=1)

    return None


def plot_mol_gradram_from_tensors(
    x_full,
    node_mask,
    mol,
    edges,
    grad_ram_weights,
    value,
    target_features,
    max_nodes,
    v=False, h=False,
    rotation=0, size=2500, title=True
):

    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # x_full: [1, n_nodes, 3] -> [n_nodes, 3]
    x = x_full[0].detach().cpu().numpy()

    # apply node mask if needed
    if node_mask is not None:
        mask = node_mask[0,:,0].detach().cpu().numpy().astype(bool)
        x = x[mask]
        grad_ram_weights = grad_ram_weights[mask]
        orig_indices = np.where(mask)[0] # keep original node indices

    # align
    x3d, Vt, xmean = align_to_xy_plane(x)
    x = x3d[:, :2]

    # rotate atoms with SAME centering+rotation as nodes
    x_atoms = (mol.get_coord() - xmean) @ Vt.T
    x_atoms = x_atoms[:, :2]

    xm = np.array([x[:, 0].max() + x[:, 0].min(),
                x[:, 1].max() + x[:, 1].min()]) / 2
    x = x - xm
    x_atoms = x_atoms - xm   # <-- important

    x, x_atoms = flip(x, x_atoms, v=v, h=h)

    # normalize weights safely
    den = np.abs(grad_ram_weights).max()
    den = den if den > 0 else 1.0
    w_scaled = grad_ram_weights / den

    ax.scatter( x[:, 0], x[:, 1], s=size, c=w_scaled, alpha=0.5, cmap='coolwarm', vmin=-1, vmax=1)

    for i in range(len(grad_ram_weights)):
        k = orig_indices[i]  # original node index
        # real node
        if k < max_nodes:
            ax.text(
                x[i, 0],
                x[i, 1],
                f"{grad_ram_weights[i]:.3f}",
                ha='center',
                va='center',
                fontsize=10,
                color='black',
                zorder=3
            )

        # orientation node
        else: # max_nodes <= i < 2 * max_nodes
            ax.text(
                x[i, 0],
                x[i, 1] - 0.3,  # vertical offset
                f"{grad_ram_weights[i]:.3f}",
                ha='center',
                va='top',
                fontsize=8,
                color='white',
                zorder=3,
                path_effects=[
                    pe.Stroke(linewidth=1.5, foreground="black"),
                    pe.Normal(),
                ]
            )


    # plot molecule
    moldraw(ax, x_atoms, mol, edges)
    if title:
        ax.set_title(f"{target_features}: {value:.3f} eV", y=0.1, pad=-25, verticalalignment="top")

    return fig

def align_to_xy_plane(x):
    xm = x.mean(axis=0)
    xc = x - xm
    _, _, Vt = np.linalg.svd(xc, full_matrices=False)
    x_rot = xc @ Vt.T
    return x_rot, Vt, xm


