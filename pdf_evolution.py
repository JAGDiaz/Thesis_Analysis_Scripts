
import os
import sys
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.animation as anime

def printProgressBar(value,label,maximum):
    n_bar = 40 #size of progress bar
    value += 1
    j= value/maximum
    sys.stdout.write('\r')
    bar = '█' * int(n_bar * j)
    bar = bar + '-' * int(n_bar * (1-j))
    sys.stdout.write(f"{label.ljust(10)} | [{bar:{n_bar}s}] {int(100 * j)}% ")
    sys.stdout.flush()
    if value == maximum:
        print()

examples_folder = os.path.join(os.getcwd(), "examples")

models = os.listdir(examples_folder)

model_folders = [os.path.join(examples_folder, model) for model in models]

for model, model_name in zip(model_folders, models):
    print(f"Generating data for {model_name}.")
    trained_folder = os.path.join(model, "trained_models")

    for dim_run in os.listdir(trained_folder):
        print(f"{dim_run}:")
        fold_components = dim_run.split('_')
        lat_dim = fold_components[-2]
        run_time = fold_components[-1]

        model_weight_folder = os.path.join(trained_folder, dim_run)
        
        pdf_dict = pkl.load(open(os.path.join(model_weight_folder, "pdf_dict.pkl"), "rb"))

        for layer in pdf_dict:
            gif_path = os.path.join(model_weight_folder, f"pdf_{layer}.gif")
            
            if os.path.exists(gif_path):
                continue

            y, x = pdf_dict[layer]

            meta = dict(title=f"pdf evolution of {layer}", artist="Matplotlib")
            writer = anime.FFMpegWriter(fps=20, metadata=meta)

            fig, ax = plt.subplots(figsize=(10,7))

            lines, = ax.plot(x, y[0], '-k')
            overshoot = .01*np.ptp(y)
            ax.set(xlim=x[[0,-1]], ylim=(0-overshoot, y.max()+overshoot))
            ax.set_xlabel("Weight", size=17.5)
            ax.set_ylabel("Density", size=17.5)
            ax.tick_params(which='both', length=9, labelsize=12.5)
            ax.set_title(f"PDF evolution of {layer.replace('_', ' ').capitalize()}, {model_name.replace('_', ' ').capitalize()}\n"
                         f"Latent Dim: {lat_dim}, Inter epoch {0:03d}", size=25)
            ax.grid(True, axis='both', which='both')

            fig.tight_layout()

            with writer.saving(fig, gif_path, 150):
                for ii in range(y.shape[0]):
                    ax.set_title(f"Density evolution of {layer.replace('_', ' ').capitalize()}, {model_name.replace('_', ' ').capitalize()}\n"
                                f"Latent Dim: {lat_dim}, Inter epoch {ii+1:03d}", size=25)
                    lines.set_data(x, y[ii])

                    writer.grab_frame()
                    
                    printProgressBar(ii, layer, y.shape[0])
            
            plt.close(fig)