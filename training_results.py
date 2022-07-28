
import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import DLDMD as dl
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
             'savefig.format':'png'}
plt.rcParams.update(**styledict)

def myactivation(x):
    return tf.keras.activations.elu(x, alpha=.01)

tf.keras.backend.set_floatx('float64')  # !! Set precision for the entire model here

examples_folder = os.path.join(os.getcwd(), "examples")

models = [file for file in os.listdir(examples_folder) if os.path.splitext(file)[1] == '']

model_folders = [os.path.join(examples_folder, model) for model in models]

for model, model_name in zip(model_folders, models):
    print(f"Generating plots for {model_name}.")
    trained_folder = os.path.join(model, "trained_models")

    feed_data = pkl.load(open(os.path.join(model, "data_val.pkl"), "rb")).numpy()
    choices = np.random.choice(np.arange(feed_data.shape[0]), size=100, replace=False)

    fig, ax = plt.subplots()
    ax.plot(*np.transpose(feed_data[choices], axes=[2,1,0]))
    ax.set(xlabel="$x$", ylabel="$y$", title=f"Validation Data")
    fig.tight_layout()
    fig.savefig(os.path.join(model, "trajectories_00.png"))
    plt.close(fig)

    feed_data = tf.cast(feed_data[choices], tf.float64)

    dim_runs = os.listdir(trained_folder)

    fug, uxes = plt.subplots(len(dim_runs)//3, 3, figsize=(30, 5*len(dim_runs)//3))
    uxes = uxes.flatten()

    for ux, dim_run in zip(uxes, dim_runs):
        print(f"{dim_run}", end="... ")
        fold_components = dim_run.split('_')
        lat_dim = fold_components[-2]
        run_time = fold_components[-1]

        model_weight_folder = os.path.join(trained_folder, dim_run, "weights_by_epoch")
        #params_folder = os.path.join(model_weight_folder, "params")
        #file_to_use = [file for file in os.listdir(params_folder) if file.endswith('.pkl')][-1]
        #print(file_to_use)
        try:
            #hyp_params = pkl.load(open(os.path.join(params_folder, file_to_use), "rb"))
            weight_file = os.path.join(model_weight_folder, [file for file in os.listdir(model_weight_folder) if file.endswith(".h5")][-1])

            hyp_params = dict()
            hyp_params['precision'] = tf.keras.backend.floatx()
            hyp_params['time_final'] = 20
            hyp_params['delta_t'] = 0.02
            hyp_params['num_time_steps'] = int(hyp_params['time_final']/hyp_params['delta_t'] + 1)
            hyp_params['num_pred_steps'] = hyp_params['num_time_steps']
            hyp_params['max_epochs'] = 1000

            # Universal network layer parameters (AE & Aux)
            hyp_params['optimizer'] = 'adam'
            hyp_params['batch_size'] = 128
            hyp_params['phys_dim'] = 2
            hyp_params['latent_dim'] = int(lat_dim)
            hyp_params['hidden_activation'] = myactivation if model_name != 'van_der_pol' else tf.keras.activations.relu
            hyp_params['bias_initializer'] = tf.keras.initializers.Zeros

            # Encoding/Decoding Layer Parameters
            hyp_params['num_en_layers'] = 3
            hyp_params['num_en_neurons'] = 128
            hyp_params['kernel_init_enc'] = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)
            hyp_params['kernel_init_dec'] = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)
            hyp_params['ae_output_activation'] = tf.keras.activations.linear

            # Loss Function Parameters
            hyp_params['a1'] = tf.constant(1, dtype=hyp_params['precision'])        # Reconstruction
            hyp_params['a2'] = tf.constant(1, dtype=hyp_params['precision'])        # DMD
            hyp_params['a3'] = tf.constant(1, dtype=hyp_params['precision'])        # Prediction
            hyp_params['a4'] = tf.constant(1e-14, dtype=hyp_params['precision'])    # L-2 on weights

            # Learning rate
            hyp_params['lr'] = 1e-3
            machine = dl.DLDMD(hyp_params)
            machine(tf.random.uniform([hyp_params['batch_size'], hyp_params['num_time_steps'], hyp_params['phys_dim']]))
            machine.load_weights(weight_file)

            result_data = machine(feed_data)

            fig, ax = plt.subplots()
            ax.plot(*np.transpose(result_data[1], axes=[2,1,0]))
            # ax.plot(*np.transpose(result_data[1], axes=[2,1,0])[:, :, 0], '.k', ms=5)
            ax.set(xlabel="$x$", ylabel="$y$", title=f"Latent Dimension: {lat_dim}")
            fig.tight_layout()
            fig.savefig(os.path.join(model, f"trajectories_{lat_dim}.png"))
            plt.close(fig)

            ux.plot(*np.transpose(result_data[1], axes=[2,1,0]))
            # ax.plot(*np.transpose(result_data[1], axes=[2,1,0])[:, :, 0], '.k', ms=5)
            ux.set(xlabel="$x$", ylabel="$y$", title=f"Latent Dimension: {lat_dim}")


            print("Done!")

        except Exception as e:
            print(e)


    fug.tight_layout()
    fug.savefig(os.path.join(model, f"experiment_trajectories.png"))
    plt.close(fug)

