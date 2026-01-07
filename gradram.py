import pickle

import dgl
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, Tensor

from data.mol import Mol


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

def plot_mol_gradram(g, mol, edges, grad_ram_weights, value,
                     target_features, v=False, h=False,
                     rotation=0, size=2500, title=True):

    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    x = g.ndata['x'].detach().cpu().numpy()
    Vt = align_to_x_plane(x)
    Vt2 = align_manual(x, rotation)
    Vt = Vt.T @ Vt2.T
    x = x[:,:2] @ Vt.T
    xm = np.array([x[:,0].max() + x[:,0].min(),x[:,1].max() + x[:,1].min()])/2
    x = x - xm

    x_atoms = mol.get_coord()[:,:2]
    x_atoms = x_atoms @ Vt.T - xm
    x, x_atoms = flip(x, x_atoms, v=v, h=h)

    w_scaled = grad_ram_weights / np.abs(grad_ram_weights).max()
    ax.scatter(x[:, 0], x[:, 1], s=size, c=w_scaled, alpha=0.5,
               cmap='coolwarm', vmin=-1, vmax=1)

    for i in range(grad_ram_weights.shape[0]):
        ax.annotate(f"{grad_ram_weights[i]:.3f}", (x[i, 0], x[i, 1]),
                    ha='center', va='center')

    # plot molecule
    moldraw(ax, x_atoms, mol, edges)
    ax.set_title(f"{target_features}: {value:.3f} eV", y=0.1, pad=-25, verticalalignment="top")
    return fig




