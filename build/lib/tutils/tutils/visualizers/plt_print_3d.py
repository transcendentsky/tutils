"""
    Print arbitary 2D Image, by Matplotlib
    For Analysis
"""

import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np


def print_2d_img(img:np.ndarray, fname="tmp.png"):
    """
        sns.heatmap():
            params: linewidth=0.5
    """
    assert type(img) == np.ndarray, f"Got {type(img)}"
    assert len(img.shape) == 2, f"Got {img.shape}"
    ax = sns.heatmap(img)
    plt.savefig(fname)
    

def print_3d_img(img:np.ndarray, fname="tmp.svg", fig_shape=(6,6)):
    """
        sns.heatmap():
            params: linewidth=0.5
    """
    assert type(img) == np.ndarray, f"Got {type(img)}"
    assert len(img.shape) == 3, f"Got {img.shape}"
    fig, axs = plt.subplots(ncols=fig_shape[0], nrows=fig_shape[1])
    # ax = sns.heatmap(img)
    vmin = img.min()
    vmax = img.max()
    
    for i in range(fig_shape[0]):
        for j in range(fig_shape[0]):
            if i*fig_shape[0] + j >= img.shape[-1]:
                break
            print(f"debug: Drawing img [{i}, {j}]")
            sns.heatmap(img[:,:,i*fig_shape[0]+j], annot=False, xticklabels=False, yticklabels=False, cbar=False, ax=axs[i,j], vmin=vmin, vmax=vmax)    
    fig.colorbar(axs[0,0].collections[0], cax=axs[0,0])
    
    #     sns.heatmap(img, annot=False, xticklabels=False, yticklabels=False, cbar=False, ax=axs[0,0], vmin=vmin)
    #     sns.heatmap(img, annot=False, xticklabels=False, yticklabels=False, cbar=False, ax=axs[1,0], vmax=vmax)
    #     fig.colorbar(axs[1,0].collections[0], cax=axs[2,0])
    plt.savefig(fname)
    # plt.show()