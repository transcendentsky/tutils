# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
# matplotlib.use('agg')
plt.style.use("ggplot")


# 画散点图
def draw_scatter(points, points2, fname="ttest.png", c="red", set_font=None, xlabel="x", ylabel="y"):
    plt.ioff()  # Turn off interactive plotting off
    if set_font is not None:
        plt.rc('font', family='Times New Roman')
    fig = plt.figure()
    parent, tail = os.path.split(fname)
    fig.suptitle(tail)
    points = points.flatten()
    points2 = points2.flatten()
    plt.scatter(points, points2, c=c, label="???")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname)
    plt.close()

# 画条形图
def draw_bar(labels, values, fname="tbar.pdf", title=None, color="red", set_font=None, xlabel="x", ylabel="y"):
    plt.ioff()
    if set_font is not None:
        plt.rc('font', family='Times New Roman')
    fig = plt.figure(figsize=(10,5))
    if title is not None:
        fig.suptitle(title)
    # ax = fig.add_axes([0,0,1,1])
    assert len(labels) == len(values)
    plt.bar(labels, values, alpha=0.7)
    plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.5)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    x_pos = [i for i, _ in enumerate(thresholds)]
    plt.xticks(x_pos, thresholds, rotation=360)
    plt.savefig(fname)
    plt.close()
