import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
styledict = {'axes.grid':True,
             'axes.grid.axis': 'both',
             'axes.grid.which': 'major',
             'xtick.labelsize':12.5,
             'xtick.major.size':9,
             'xtick.major.width':1,
             'ytick.labelsize':12.5,
             'ytick.major.size':9,
             'ytick.major.width':1,
             'legend.framealpha':1,
             'legend.fontsize':12.5,
             'axes.labelsize':17.5,
             'axes.titlesize':25,
             'axes.linewidth':2,
             'figure.figsize':(10,5),
             'figure.titlesize':25,
             'savefig.format':'png',
             'text.usetex':True}
plt.rcParams.update(**styledict)


examples_folder = os.path.join(os.getcwd(), "examples")
models = os.listdir(examples_folder)
model_folders = [os.path.join(examples_folder, model, "trained_models") for model in models]

cutoffs = {'duffing': 5, 'duffing_sd': 5, 'van_der_pol': 9}

for model_folder, model_name in zip(model_folders, models):

    # [os.remove(os.path.join(examples_folder, model_name, file)) for file in os.listdir(os.path.join(examples_folder, model_name)) if file.endswith('.png')]

    individual_runs = os.listdir(model_folder)[:cutoffs[model_name]]
    individual_runs_folder = [os.path.join(model_folder, individual_run) for individual_run in individual_runs]
    color_map = plt.cm.get_cmap('tab20b', len(individual_runs))
    colors = color_map([qq for qq in range(len(individual_runs))])

    lat_dims = []

    fig, ax2 = plt.subplots()#, sharex='col')

    for color, dim_run, dim_run_folder in zip(colors, individual_runs, individual_runs_folder):

        loss_file = os.path.join(dim_run_folder, "losses_by_epoch.csv")
        if not os.path.exists(loss_file):
            continue
        data_frame = pd.read_csv(loss_file, index_col=0)

        fold_components = dim_run.split('_')
        lat_dim = fold_components[-2]
        run_time = fold_components[-1]

        lat_dims.append(lat_dim)
        epoch = data_frame.index.values

        loss_data = data_frame[data_frame.columns[0]].values

        # ax1.plot(epoch, loss_data, '-', label=lat_dim)
        ax2.plot(epoch, savgol_filter(loss_data, 31, 5), '-', label=lat_dim)


    #ax1.set_yscale('log')
    # ax1.grid(True, which='both')
    # ax1.set_ylabel(r"$\log_{10}$ Loss", size=17.5)
    # ax1.tick_params(axis='y', labelsize=12.5, length=9)
    # ax1.tick_params(axis='x', length=0)
    # ax1.legend(loc='best', fontsize=10, ncol=5)        

    # ax2.set_yscale('log')
    ax2.grid(True, which='both')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(r"$\log_{10}$ Loss")
    ax2.tick_params(axis='both')
    ax2.legend(loc='best', fontsize=10, ncol=5)

    fig.suptitle(f"Model: {model_name.replace('_', ' ').capitalize()}", size=25)


    fig.tight_layout()
    fig.savefig(os.path.join(examples_folder, model_name, f"loss_plots.png"))