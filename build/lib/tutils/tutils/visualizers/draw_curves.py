import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
# matplotlib.use('agg')


def draw_heatmap(points: np.ndarray, points2, fname="./tmp/heatmap.png"):
    # print(count1, count2) # 68 88
    # points = np.reshape(points, (count1, count2))
    plt.rc('font', family='Times New Roman')
    fig = plt.figure(figsize=(25, 20))
    parent, tail = os.path.split(fname)
    fig.suptitle(tail)
    ax1 = fig.add_subplot(221)
    sns_plot = sns.heatmap(points)
    # fig, ax = plt.subplots()
    ax2 = fig.add_subplot(222)
    sns_plot = sns.heatmap(points2)
    fig.savefig(fname)
    # fig.show()
    plt.close()


def draw_scatter(points, points2, fname="./tmp/scatter.png", c="red", set_font=None, xlabel="x", ylabel="y"):
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


# ----------------------------------------------------------------
# for debug
def draw_scripts():
    from numpy import random
    R = random.randn(11, 11)
    R2 = random.randn(11, 11)
    draw_heatmap(R, R2)


def draw_saved_scatter(dir_path):
    points1_group = []
    points2_group = []
    for i in tqdm(range(149)):
        fname = f"euc_value_image_{i}.npy"
        data_euc = np.load(dir_path + fname)
        fname = f"cos_value_image_{i}.npy"
        data_cos = np.load(dir_path + fname)
        data_cos = data_cos  # for better visualization

        points1 = []
        points2 = []
        for id_landmark in range(19):
            idx = np.argmax(data_cos[id_landmark])
            points1 += [data_cos[id_landmark][idx]]
            points2 += [data_euc[id_landmark]]

        points1_group.append(points1)
        points2_group.append(points2)

    for i in range(19):
        re_points1 = [x[i] for x in points1_group]
        re_points2 = [x[i] for x in points2_group]
        draw_scatter(
            np.array(re_points1), np.array(re_points2),
            fname=f"output/scatter/scatter_lanmark_{i}.png")


def draw_saved(dir_path):
    for i in tqdm(range(149)):
        fname = f"euc_value_image_{i}.npy"
        data_euc = np.load(dir_path + fname)
        fname = f"cos_value_image_{i}.npy"
        data_cos = np.load(dir_path + fname)
        data_cos = 1 - data_cos  # for better visualization

        for id_landmark in range(19):
            draw_heatmap(
                np.reshape(data_euc[id_landmark], (68, 88)),
                np.reshape(data_cos[id_landmark], (68, 88)),
                fname=f"output/img/mark{id_landmark}_im{i}.png")
