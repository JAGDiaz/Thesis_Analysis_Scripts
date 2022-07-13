
import os
import sys
import h5py 
import numpy as np
import pandas as pd
from KDEpy import FFTKDE
import scipy.integrate as integrate
import pickle as pkl

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

    for ii, layer in enumerate(weight_diffs.T):
        
        layer = np.array([ii for ii in layer])
        overshoot = .01*np.ptp(layer)
        layer_support = np.linspace(layer.min()-overshoot, layer.max()+overshoot, 10001)
        layer_pdfs = [None]*layer.shape[0]
        layer_bandwidths = [None]*layer.shape[0]

        for jj, data in enumerate(layer):

            estimator = FFTKDE(kernel=kde_kernel, bw=kde_bw).fit(data)
            layer_bandwidths[jj] = estimator.bw
            pdf = estimator.evaluate(layer_support)
            simpson_area = integrate.simpson(pdf, layer_support)

            layer_pdfs[jj] = pdf/simpson_area
        
        pdfs[ii] = layer_pdfs
        supports[ii] = layer_support
        bands[ii] = layer_bandwidths

    return np.array(pdfs), np.array(supports), np.array(bands)

examples_folder = os.path.join(os.getcwd(), "examples")

models = os.listdir(examples_folder)

model_folders = [os.path.join(examples_folder, model) for model in models]

for model, model_name in zip(model_folders, models):
    print(f"Generating data for {model_name}.")
    trained_folder = os.path.join(model, "trained_models")

    for dim_run in os.listdir(trained_folder):
        
        fold_components = dim_run.split('_')
        lat_dim = fold_components[-2]
        run_time = fold_components[-1]

        model_weight_folder = os.path.join(trained_folder, dim_run)
        weight_files = [os.path.join(model_weight_folder, file) for file in os.listdir(model_weight_folder) if file.endswith('.h5')]

        if len(weight_files) != 1000 and not os.path.exists(os.path.join(model_weight_folder, 'divergence_results.csv')):
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
        printProgressBar(0, dim_run, 4)

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


        enc_pdfs, enc_supports, enc_bws = pdf_creator(enc_weights_diff)
        dec_pdfs, dec_supports, dec_bws = pdf_creator(dec_weights_diff)
        printProgressBar(1, dim_run, 4)

        full_weight_diff_dict = {**{lab:layer for lab, layer in zip(enc_labels, enc_weights_diff.T)}, 
                                 **{lab:layer for lab, layer in zip(dec_labels, dec_weights_diff.T)}}
        pkl.dump(full_weight_diff_dict, open(os.path.join(model_weight_folder, "all_weights_diff_array_dict.pkl"),'wb'))
        del dec_weights_diff, enc_weights_diff, full_weight_diff_dict

        pdf_dict = {**{lab:(layer, support) for lab, support, layer in zip(enc_labels, enc_supports, enc_pdfs)}, 
                    **{lab:(layer, support) for lab, support, layer in zip(dec_labels, dec_supports, dec_pdfs)}}
        pkl.dump(pdf_dict, open(os.path.join(model_weight_folder, "pdf_dict.pkl"),'wb'))

        enc_divergences = get_diveregences(enc_pdfs, enc_supports)
        dec_divergences = get_diveregences(dec_pdfs, dec_supports)        
        printProgressBar(2, dim_run, 4)
        del enc_supports, dec_supports, enc_pdfs, dec_pdfs

        all_divergences = np.concatenate([enc_divergences, dec_divergences], axis=0).T
        all_bws = np.concatenate([enc_bws, dec_bws], axis=0).T
        column_labels = enc_labels+dec_labels


        data_frame = pd.DataFrame(data=all_divergences, columns=column_labels, index=np.arange(all_divergences.shape[0])+1)
        data_frame.to_csv(os.path.join(model_weight_folder, "divergence_results.csv"))

        data_frame = pd.DataFrame(data=all_bws, columns=column_labels, index=np.arange(all_bws.shape[0])+1)
        data_frame.to_csv(os.path.join(model_weight_folder, "bandwidth_results.csv"))

        printProgressBar(3, dim_run, 4)
        del enc_divergences, dec_divergences, all_divergences
        del enc_bws, dec_bws, all_bws
        del data_frame