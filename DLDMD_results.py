
import os
import sys
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.animation as anime
import tensorflow as tf
from tensorflow import keras
import DLDMD as dl

tf.keras.backend.set_floatx('float64')  # !! Set precision for the entire model here

examples_folder = os.path.join(os.getcwd(), "examples")

models = os.listdir(examples_folder)

model_folders = [os.path.join(examples_folder, model) for model in models]

for model, model_name in zip(model_folders, models):
    print(f"Generating plots for {model_name}.")
    trained_folder = os.path.join(model, "trained_models")

    feed_data = pkl.load(open(os.path.join(model, "data_val.pkl"), "rb")).numpy()
    choices = np.random.choice(np.arange(feed_data.shape[0]), size=100, replace=False)

    fig, ax = plt.subplots(figsize=(10,7))
    ax.plot(*np.transpose(feed_data[choices], axes=[2,1,0]))
    fig.tight_layout()
    fig.savefig(os.path.join(model, "trajectories_og.png"))
    plt.close(fig)

    feed_data = tf.cast(feed_data[choices], tf.float64)


    for dim_run in os.listdir(trained_folder):
        print(f"{dim_run}:")
        fold_components = dim_run.split('_')
        lat_dim = fold_components[-2]
        run_time = fold_components[-1]

        model_weight_folder = os.path.join(trained_folder, dim_run)
        params_folder = os.path.join(model_weight_folder, "params")
        file_to_use = [file for file in os.listdir(params_folder) if file.endswith('.pkl')][-1]
        #print(file_to_use)
        try:
            hyp_params = pkl.load(open(os.path.join(params_folder, file_to_use), "rb"))
            weight_file = os.path.join(model_weight_folder, [file for file in os.listdir(model_weight_folder) if file.endswith(".h5")][-1])

            machine = dl.DLDMD(hyp_params)
            machine(tf.random.uniform([hyp_params['batch_size'], hyp_params['num_time_steps'], hyp_params['phys_dim']]))
            machine.load_weights(weight_file)

            result_data = machine(feed_data)

            fig, ax = plt.subplots(figsize=(10,7))
            ax.plot(*np.transpose(result_data[1], axes=[2,1,0]))
            fig.tight_layout()
            fig.savefig(os.path.join(model, f"trajectories_{lat_dim}.png"))
            plt.close(fig)

        except Exception as e:
            print(e)



