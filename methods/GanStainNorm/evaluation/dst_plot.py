"""Plots different distances using matplotlib"""

from argparse import ArgumentParser

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_diff_colors(org, tgt, dst=True):
    """Generates a color array based on positive/negative difference"""
    diff = org - tgt
    colors = [[0., 0.5, 0., 1.] if x > 0 else [0.5, 0.0, 0., 1.] for x in diff]\
        if dst else\
            [[0.5, 0.0, 0., 1.] if x > 0 else [0., 0.5, 0., 1.] for x in diff]
    return colors

def plot_distances(
        main_pkl:str,
        sec_pkl:str,
        main_title: str,
        sec_title: str,
        plot_title: str
    ):
    """Plot distances"""
    #main_pkl = 'original_b.pkl'
    #sec_pkl = 'reinhard_b.pkl'

    main_df = pd.read_pickle(main_pkl)
    sec_df = pd.read_pickle(sec_pkl)

    main_df = main_df[main_df['sample_name'].isin(list(sec_df['sample_name']))]

    main_df.sort_values(by=['sample_name'], inplace=True)
    sec_df.sort_values(by=['sample_name'], inplace=True)

    columns = list(main_df.columns)
    columns.remove('sample_name')
    ax_indexes = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    samples = [i for i in range(len(sec_df))]

    fig, axs = plt.subplots(2, 3, figsize=(14, 8))

    for i, col in enumerate(columns):
        dst = False if col in ['intersection', 'pearson_r'] else True
        main_col = main_df[col].tolist()
        sec_col = sec_df[col].tolist()
        diff_colors = get_diff_colors(np.array(main_col), np.array(sec_col), dst)
        axs[ax_indexes[i]].plot(samples, main_col, 'b.', label=main_title)
        axs[ax_indexes[i]].plot(samples,  sec_col, 'm.', label=sec_title)
        axs[ax_indexes[i]].vlines(
            samples, main_col, sec_col, colors=diff_colors)
        axs[ax_indexes[i]].set_title(col.replace('_', ' ').capitalize())
        axs[ax_indexes[i]].legend()
    #fig.tight_layout()
    fig.suptitle(plot_title)
    plt.show()

def main():
    """The main execution function"""

    parser = ArgumentParser()
    parser.add_argument(
        '--main_pkl', help='Full path to main evaluation pickle file', type=str, required=True)
    parser.add_argument(
        '--sec_pkl', help='Full path to secondary evaluation pickle file', type=str, required=True)
    parser.add_argument(
        '--main_title', help='Main data/method title', type=str, required=True)
    parser.add_argument(
        '--sec_title', help='Secondary data/method title', type=str, required=True)
    parser.add_argument(
        '--plot_title', help='Overall plot title', type=str, required=True)

    param = parser.parse_args()

    plot_distances(
        main_pkl=param.main_pkl,
        sec_pkl=param.sec_pkl,
        main_title=param.main_title,
        sec_title=param.sec_title,
        plot_title=param.plot_title
    )

if __name__ == '__main__':
    main()
    