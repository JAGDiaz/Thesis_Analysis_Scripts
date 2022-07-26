
import os
import sys
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.animation as anime

def computeArea(pos):
    x, y = (zip(*pos))
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

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

models = [file for file in os.listdir(examples_folder) if os.path.splitext(file)[1] == '']

model_folders = [os.path.join(examples_folder, model) for model in models]

for model, model_name in zip(model_folders, models):
    print(f"Generating data for {model_name}.")
    trained_folder = os.path.join(model, "trained_models")

    for dim_run in os.listdir(trained_folder):
        print(f"{dim_run}")
        fold_components = dim_run.split('_')
        lat_dim = fold_components[-2]
        run_time = fold_components[-1]

        model_dim_folder = os.path.join(trained_folder, dim_run)

        bandwidth_dirs = [os.path.join(model_dim_folder, file) for file in os.listdir(model_dim_folder) if ('bandwidth' in file and not file.endswith('.csv'))]
        
        for bandwidth_dir in bandwidth_dirs:
            bandie = bandwidth_dir.split('_')[-1]

            try:
                pdf_dict = pkl.load(open(os.path.join(bandwidth_dir, "pdf_dict.pkl"), "rb"))
            except Exception as e:
                print(e)
                continue
            
            for layer in pdf_dict:
                gif_path = os.path.join(bandwidth_dir, f"pdf_{layer}.gif")
                
                if os.path.exists(gif_path):
                    continue

                y, x = pdf_dict[layer]

                meta = dict(title=f"pdf evolution of {layer}", artist="Matplotlib")
                writer = anime.FFMpegWriter(fps=20, metadata=meta)

                fig, ax = plt.subplots(figsize=(10,7))

                lines, = ax.plot(x, y[0], '-k')
                overshoot = .01*np.ptp(y)
                ax.set(xlim=x[[0,-1]], ylim=(0-overshoot, y.max()+overshoot))
                # if y.max() > 10000:
                    # ax.set_yscale('log')
                    # ax.set_ylim(1e-3 , y.max() + overshoot)
                ax.set_xlabel("Weight", size=17.5)
                ax.set_ylabel("Density", size=17.5)
                ax.tick_params(which='both', length=9, labelsize=12.5)
                ax.set_title(f"{model_name.replace('_', ' ').capitalize()}, {layer.replace('_', ' ').capitalize()}, "
                              f"Latent Dim: {lat_dim},\n Inter epoch {0:03d}, bw: {bandie}, Int Area: {1:1.6f}", size=25)
                ax.grid(True, axis='both', which='both')

                fig.tight_layout()

                with writer.saving(fig, gif_path, 100):
                    for ii in range(y.shape[0]):

                        poly_coll = plt.fill_between(x, y[ii], color='#1f77b4')

                        path = poly_coll.get_paths()[0]
                        verts = path.vertices
                        area = computeArea(verts)
                        ax.set_title(f"{model_name.replace('_', ' ').capitalize()}, {layer.replace('_', ' ').capitalize()}, "
                                     f"Latent Dim: {lat_dim},\n Inter epoch {ii+1:03d}, bw: {bandie}, Int Area: {area:1.6f}", size=25)

                        lines.set_data(x, y[ii])

                        writer.grab_frame()
                        poly_coll.remove()
                        printProgressBar(ii, f"{layer}, bw: {bandie}", y.shape[0])
                
                plt.close(fig)
                del fig

            del pdf_dict