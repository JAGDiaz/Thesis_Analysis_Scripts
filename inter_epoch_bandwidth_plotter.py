import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter



examples_folder = os.path.join(os.getcwd(), "examples")
models = os.listdir(examples_folder)
model_folders = [os.path.join(examples_folder, model, "trained_models") for model in models]

cutoffs = {'duffing': 5, 'duffing_sd': 5, 'van_der_pol': 9}

for model_folder, model_name in zip(model_folders, models):

    # [os.remove(os.path.join(examples_folder, model_name, file)) for file in os.listdir(os.path.join(examples_folder, model_name)) if file.endswith('.png')]

    individual_runs = os.listdir(model_folder)[:cutoffs[model_name]]
    individual_runs_folder = [os.path.join(model_folder, individual_run) for individual_run in individual_runs]

    average_band_dict = dict()
    std_dev_band_dict = dict()

    fug, ux = plt.subplots(figsize=(10,7))

    for dim_run, dim_run_folder in zip(individual_runs, individual_runs_folder):

        bandwidth_file = os.path.join(dim_run_folder, "bandwidth_results.csv")

        if not os.path.exists(bandwidth_file):
            continue

        fold_components = dim_run.split('_')
        lat_dim = fold_components[-2]
        run_time = fold_components[-1]
        

        band_frame = pd.read_csv(bandwidth_file, index_col=0)
        all_band_data = band_frame.values
        
        average_band_dict[lat_dim] = np.mean(all_band_data, axis=1)
        std_dev_band_dict[lat_dim] = np.std(all_band_data, axis=1)

        inter_epoch = band_frame.index.values

        #ux[0].plot(inter_epoch, average_band_dict[lat_dim], label=lat_dim)
        ux.plot(inter_epoch, savgol_filter(average_band_dict[lat_dim], 31, 5), label=lat_dim)

        fig, ax2 = plt.subplots(figsize=(10, 7))

        for column in band_frame.columns:
            data = band_frame[column].values

            # ax1.plot(inter_epoch, data, '-', label=column)
            ax2.plot(inter_epoch, savgol_filter(data, 31, 5), '-', label=column)

        # ax1.set_yscale('log')
        # ax1.grid(True, which='both')
        # ax1.set_xlabel("Inter Epoch", size=20)
        # ax1.set_ylabel("Bandwidth", size=17.5)
        # ax1.tick_params(axis='y', labelsize=12.5, length=9)
        # ax1.tick_params(axis='x', length=0)
        # ax1.legend(loc='best', fontsize=10, ncol=5)        

        ax2.set_yscale('log')
        ax2.grid(True, which='both')
        ax2.set_xlabel("Inter Epoch", size=17.5)
        ax2.set_ylabel("Bandwidth", size=17.5)
        ax2.tick_params(axis='both', labelsize=12.5, length=9)
        ax2.legend(loc='best', fontsize=10, ncol=5)

        fig.suptitle(f"Model: {model_name}, Latent Dim: {lat_dim}, Time: {run_time}", size=22.5)


        fig.tight_layout()
        fig.savefig(os.path.join(examples_folder, model_name, f"bandwidth_plots_{lat_dim}_{run_time}.png"))

        plt.close(fig)
    
    # for _ in ux:
        # _.grid(True, which='both')
        # _.tick_params(axis='y', labelsize=12.5, length=9)
        # _.set_ylabel("Average Bandwidth", size=17.5)
        # _.set_yscale('log')
        # _.legend(loc='best', fontsize=10, ncol=5)

    ux.grid(True, which='both')
    ux.tick_params(axis='y', labelsize=12.5, length=9)
    ux.set_ylabel("Average Bandwidth", size=17.5)
    ux.set_yscale('log')
    ux.legend(loc='best', fontsize=10, ncol=5)

    # ux[0].set_title(f"Model: {model_name.replace('_',' ').capitalize()}", size=22.5)
    # ux[0].tick_params(axis='x', length=0, labelsize=0)
    ux.set_xlabel("Inter Epoch", size=17.5)


    fug.tight_layout()
    fug.savefig(os.path.join(examples_folder, model_name, f"bandwidth_averages_plot.png"))
    
    plt.close(fug)
