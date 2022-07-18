
import os
import sys
import h5py 
import numpy as np
import pandas as pd
from KDEpy import FFTKDE
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pickle as pkl
from numba import vectorize
import warnings
warnings.filterwarnings("ignore")

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


def walk_h5(file_handle, space="", num_spaces=2):
    for key in file_handle.keys():
        print(space + key)
        try:
            walk_h5(file_handle[key], space=space + " "*num_spaces, num_spaces=num_spaces)
        except:
            pass

def continuous_KLD(p, q, x, tol=1e-12):
    integrand = np.where(p > tol, p * np.log(p / q), 0)
    return integrate.simpson(integrand, x)

def sym_KLD(p, q, x, vep=1e-12):
    return .5*(continuous_KLD(p, q, x, vep) + continuous_KLD(q, p, x, vep))

@vectorize
def epa_kernel(x):
    if np.abs(x) < 1:
        return .75*(1-x*x)
    else:
        return 0

def density_maker(x_vals, h, kernel=epa_kernel):
    n = x_vals.size
    def density(x):
        X, Xi = np.meshgrid(x, x_vals)
        values = kernel((X - Xi)/h)
        
        return np.sum(values, axis=0)/(n*h)
    return density

def get_diveregences(data_set, supports):

    divs = np.zeros(shape=(data_set.shape[0], data_set.shape[1]-1))
    
    for ii, (layer, support) in enumerate(zip(data_set, supports)):

        for jj in range(layer.shape[0]-1):
            divs[ii, jj] = sym_KLD(layer[jj], layer[jj+1], support)

    return divs

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

def pdf_creator(weight_diffs, kde_kernel='epa', kde_bw='ISJ'):

    pdfs = [None]*weight_diffs.shape[1]
    supports = [None]*weight_diffs.shape[1]
    bands = [None]*weight_diffs.shape[1]
    simps_areas = [None]*weight_diffs.shape[1]


    for ii, layer in enumerate(weight_diffs.T):
        
        layer = np.array([ii for ii in layer])
        full_overshoot = .05*np.ptp(layer)
        full_support = np.linspace(layer.min()-full_overshoot, layer.max()+full_overshoot, 5001)
        layer_pdfs = [None]*layer.shape[0]
        layer_bandwidths = [None]*layer.shape[0]
        layer_simps_area = [None]*layer.shape[0]

        for jj, data in enumerate(layer):
            
            #layer_overshoot = .01*np.ptp(data)
            #layer_support = np.linspace(data.min()-layer_overshoot, data.max()+layer_overshoot, full_support.size)
            estimator = FFTKDE(kernel=kde_kernel, bw=kde_bw).fit(data)
            layer_bandwidths[jj] = estimator.bw
            layer_support, pdf = estimator.evaluate(1001)
            layer_func = density_maker(layer_support, estimator.bw)
            #simpson_area = integrate.simpson(pdf, layer_support)
            layer_simps_area[jj] = integrate.quadrature(layer_func, *full_support[[0,-1]])[0]


            layer_pdfs[jj] = layer_func(full_support)
        
        pdfs[ii] = layer_pdfs
        supports[ii] = full_support
        bands[ii] = layer_bandwidths
        simps_areas[ii] = layer_simps_area

    return np.array(pdfs), np.array(supports), np.array(bands), np.array(simps_areas)

examples_folder = os.path.join(os.getcwd(), "examples")

models = [file for file in os.listdir(examples_folder) if os.path.splitext(file)[1] == '']

model_folders = [os.path.join(examples_folder, model) for model in models]

fug, uxes = plt.subplots(1,len(model_folders), figsize=(5*len(model_folders),5), sharey='row')

for model, model_name, ux in zip(model_folders, models, uxes):
    print(f"Generating data for {model_name}.")
    trained_folder = os.path.join(model, "trained_models")

    disties = []

    for dim_run in os.listdir(trained_folder):
        
        fold_components = dim_run.split('_')
        lat_dim = fold_components[-2]
        run_time = fold_components[-1]

        model_weight_folder = os.path.join(trained_folder, dim_run)
        weight_files = [os.path.join(model_weight_folder, file) for file in os.listdir(model_weight_folder) if file.endswith('.h5')]

        if len(weight_files) != 1000:# or os.path.exists(os.path.join(model_weight_folder, 'divergence_results.csv')):
            continue

        enc_weights = [None]*len(weight_files)
        dec_weights = [None]*len(weight_files)

        for ii, weight_file in enumerate(weight_files):

            h5_file = h5py.File(weight_file, 'r')
            encoder_layers = h5_file['encoder']
            enc_labels = list(encoder_layers.keys())
            enc_labels.insert(0, enc_labels.pop(-2))
            enc_weights[ii] = [np.concatenate([encoder_layers[key]['kernel:0'][:].flatten(), encoder_layers[key]['bias:0'][:].flatten()]) for key in enc_labels]

            decoder_layers = h5_file['decoder']
            dec_labels = list(decoder_layers.keys())
            dec_labels.insert(0, dec_labels.pop(-2))
            dec_weights[ii] = [np.concatenate([decoder_layers[key]['kernel:0'][:].flatten(), decoder_layers[key]['bias:0'][:].flatten()]) for key in dec_labels]
            h5_file.close()

        del h5_file
        printProgressBar(0, dim_run, 3)

        enc_weights = np.array(enc_weights, dtype=object)
        dec_weights = np.array(dec_weights, dtype=object)


        enc_weights_diff = np.diff(enc_weights, axis=0)
        dec_weights_diff = np.diff(dec_weights, axis=0)

        full_weight_dict = {**{lab:layer for lab, layer in zip(enc_labels, enc_weights.T)}, 
                            **{lab:layer for lab, layer in zip(dec_labels, dec_weights.T)}}
        pkl.dump(full_weight_dict, open(os.path.join(model_weight_folder, "all_weights_array_dict.pkl"),'wb'))
        del enc_weights
        del dec_weights
        del full_weight_dict


        enc_pdfs, enc_supports, enc_bws, enc_simps = pdf_creator(enc_weights_diff)
        dec_pdfs, dec_supports, dec_bws, dec_simps = pdf_creator(dec_weights_diff)
        disties.append([enc_simps, dec_simps])
        printProgressBar(1, dim_run, 3)

        fig, axes = plt.subplots(1,2, sharey='row')
        axes[0].hist(enc_simps.flatten(), bins='auto', density=True)
        axes[1].hist(dec_simps.flatten(), bins='auto', density=True)
        axes[0].set_title("Encoder")
        axes[1].set_title("Decoder") 
        axes[0].set_xlabel("Weights")
        axes[1].set_xlabel("Weights")  
        axes[0].set_ylabel("Density")
        fig.suptitle(f"Model: {model_name.replace('_',' ').capitalize()}, Lat Dim: {lat_dim}")
        fig.tight_layout()
        fig.savefig(os.path.join(model, f"area_deviation_{lat_dim}.png"))
        plt.close(fig)

        full_weight_diff_dict = {**{lab:layer for lab, layer in zip(enc_labels, enc_weights_diff.T)}, 
                                 **{lab:layer for lab, layer in zip(dec_labels, dec_weights_diff.T)}}
        pkl.dump(full_weight_diff_dict, open(os.path.join(model_weight_folder, "all_weights_diff_array_dict.pkl"),'wb'))
        del dec_weights_diff, enc_weights_diff, full_weight_diff_dict

        pdf_dict = {**{lab:(layer, support) for lab, support, layer in zip(enc_labels, enc_supports, enc_pdfs)}, 
                    **{lab:(layer, support) for lab, support, layer in zip(dec_labels, dec_supports, dec_pdfs)}}
        pkl.dump(pdf_dict, open(os.path.join(model_weight_folder, "pdf_dict.pkl"),'wb'))

        enc_divergences = get_diveregences(enc_pdfs, enc_supports)
        dec_divergences = get_diveregences(dec_pdfs, dec_supports)        
        del enc_supports, dec_supports, enc_pdfs, dec_pdfs

        all_divergences = np.concatenate([enc_divergences, dec_divergences], axis=0).T
        all_bws = np.concatenate([enc_bws, dec_bws], axis=0).T
        column_labels = enc_labels+dec_labels


        data_frame = pd.DataFrame(data=all_divergences, columns=column_labels, index=np.arange(all_divergences.shape[0])+1)
        data_frame.to_csv(os.path.join(model_weight_folder, "divergence_results.csv"))

        data_frame = pd.DataFrame(data=all_bws, columns=column_labels, index=np.arange(all_bws.shape[0])+1)
        data_frame.to_csv(os.path.join(model_weight_folder, "bandwidth_results.csv"))

        printProgressBar(2, dim_run, 3)
        del enc_divergences, dec_divergences, all_divergences
        del enc_bws, dec_bws, all_bws
        del data_frame
    
    disties = np.array(disties)

    fig, ax = plt.subplots()

    ax.hist(disties.flatten(), bins='auto', density=True)
    ax.set(xlabel="Weights", ylabel="Density", title=f"Area Deviation for {model_name.replace('_',' ').capitalize()}")
    
    fig.tight_layout()
    fig.savefig(os.path.join(model, f"area_deviation.png"))
    plt.close(fig)
    
    ux.hist(disties.flatten(), bins='auto', density=True, label=model_name.replace('_',' ').capitalize())
    ux.set(xlabel="Integrated Area", xlim=(0,2), title=model_name.replace('_',' ').capitalize())
    del disties

uxes[0].set_ylabel("Density")
uxes[-1].tick_params(axis='y', which='both', length=0)

fug.tight_layout()
fug.savefig(os.path.join(examples_folder, "area_devs.png"))
plt.close(fug)