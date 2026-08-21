"""Plot violin plots for model comparisons"""

from argparse import ArgumentParser
from os.path import join
from typing import Sequence
from os.path import isdir
from os import listdir
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import random
from os import makedirs

random.seed(1886)
def compare_experiments(
        exp_root: str,
        output_dir: str,
        experiments: Sequence[str],
    ):
    exp_df_dict = {}
    
    tester = True
    for exp in experiments:
        eval_dir = join(exp_root, exp, 'evaluation')
        epoch = [epc for epc in listdir(eval_dir) if isdir(join(eval_dir, epc))]
        fid_df = pd.read_pickle(join(eval_dir, epoch[0], 'fid.pkl'))
        dst_df = pd.read_pickle(join(eval_dir, epoch[0], 'results.pkl'))
        ssim_df = pd.read_pickle(join(eval_dir, epoch[0], 'ssim.pkl'))

        if tester:
            tester = False
            ref_df = fid_df.set_index(
                'sample_name').join(
                    dst_df.set_index('sample_name')).join(
                    ssim_df.set_index('sample_name'))
        temp_df = fid_df.set_index(
        'sample_name').join(
            dst_df.set_index('sample_name')).join(
            ssim_df.set_index('sample_name'))
        exp_df_dict[exp] = temp_df.filter(
            items=ref_df.index.tolist(), axis=0)
        #print(exp_df_dict[exp].head(10))
    
    experiments.append('Original')
    labels = ['' for _ in np.arange(0, len(experiments)*4+3)]
    for i, j in enumerate(range(3, len(labels), 4)):
        labels[j] = experiments[i]

    tissue_type = f'tissue_{ref_df.index[0].split("_")[0]}'
    organ_dict = {'tissue_a': 'Skin', 'tissue_b': 'Kidney', 'tissue_c': 'Colon'}
    base_result_dir = '/home/local-admin/drive/projects/staining-qc/results/'
    output_dir = join(output_dir, tissue_type)
    makedirs(output_dir, exist_ok=True)

    fid_df = pd.read_pickle(join(base_result_dir, tissue_type, 'fid.pkl'))
    dst_df = pd.read_pickle(join(base_result_dir, tissue_type, 'results.pkl'))
    #ssim_df = pd.read_pickle(join(base_result_dir, tissue_type, 'ssim.pkl'))

    temp_df = fid_df.set_index(
        'sample_name').join(
        dst_df.set_index('sample_name'))#.join(
        #ssim_df.set_index('sample_name'))
    exp_df_dict['Original'] = temp_df.filter(
        items=ref_df.index.tolist(), axis=0)

    eval_metrics = list(ref_df.columns)

    
    plt.rcParams.update({'font.size': 3.5
    })
    colors = []
    for met in eval_metrics:
        i = 0
        data = np.zeros((len(experiments), len(exp_df_dict[exp])))
        for exp, df in exp_df_dict.items():
            try:
                data[i, :] = df[met].to_numpy()
                i += 1
            except:
                data = data[:-1,:]
                labels = labels[:-4]
                experiments.remove('Original')
            
        
        _, axes = plt.subplots(figsize=(7, 4), dpi=300)
        for axis in ['top', 'bottom', 'left', 'right']:
            axes.spines[axis].set_linewidth(0.125)  # change width
            axes.spines[axis].set_color('gray')

        positions  = [i for i in range(3, len(experiments)*4+3, 4)]
        trans_data = data.T

        vln_plot = axes.violinplot(
            trans_data, positions, widths=3, showmeans=True, showmedians=False, showextrema=True)
        
        if not colors:
            counter = 0
            while counter < len(experiments):
                color = random_color_generator()
                if color in colors:
                    print(color)
                    continue
                colors.append(color)
                counter += 1
        
        for body, clr in zip(vln_plot['bodies'], colors):#['#B73652', '#FDB168', '#6A90C3']):
            body.set_facecolor(clr)
            body.set_alpha(0.7)

        set_axis_style(axes, labels)
        #ax.set(xlim=(0, 9), xticks=np.arange(1, 9),
        #    ylim=(0, 100), yticks=np.arange(1, 100))
        plt.title(f'{organ_dict[tissue_type]} {met}')
        plt.grid(True, linewidth=0.25)
        #plt.show()
        plt.savefig(join(output_dir, f'violin_{met}.png'))
        plt.close()


def random_color_generator():
    return random.choice(list(mcolors.CSS4_COLORS.keys()))


def set_axis_style(axs, labels):
    "Set axis style as per number of distributions to be compared"
    axs.xaxis.set_tick_params(direction='out')
    axs.xaxis.set_ticks_position('bottom')
    axs.set_xticks(np.arange(0, len(labels)))#, labels=labels)
    ref_labels = []
    resnet = True
    for label in labels:
        if 'Exp' in label:
            if 'CycleGan' in label:
                if resnet:
                    lbl = f'{label.split("-")[1]}(RN)'
                    resnet = False
                else:
                    lbl = f'{label.split("-")[1]}(UN)'
            else:
                lbl = label.split("-")[1]
            ref_labels.append(lbl)
        elif label == 'GroundTruth':
            ref_labels.append('GroundTruth')
        else:
            ref_labels.append('')
    axs.set_xticklabels(ref_labels)


def main():
    """Takes aguements and initiate violin plotting"""

    parser = ArgumentParser()
    parser.add_argument(
        '--exp_root', help='Main training pickles directory',
        type=str, required=True)
    parser.add_argument(
        '--output_dir', help='Output directory for comparison violins',
        type=str, required=True
    )
    parser.add_argument(
        '--experiments', nargs='+', help='Experiments to be compared', 
        required=True
    )
    param = parser.parse_args()
    compare_experiments(
        exp_root=param.exp_root,
        output_dir=param.output_dir,
        experiments=param.experiments
    )
    
if __name__ == '__main__':
    main()