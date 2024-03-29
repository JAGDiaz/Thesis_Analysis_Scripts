
from hashlib import new
import os
import sys
import h5py 
import numpy as np
import pandas as pd
from KDEpy import TreeKDE, NaiveKDE, FFTKDE
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pickle as pkl
from numba import vectorize, njit

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

def discrete_KLD(p, q, dx, tol=1e-12):
    return np.sum(np.where(p > tol, p * np.log(p / q), 0))*dx

@njit
def dis_KLD_jit(p, q, dx, tol=1e-12):

    where = p > tol
    new_p = p[where]
    new_q = q[where]
    div = 0
    for ii in range(new_p.size):
            div += new_p[ii]*np.log(new_p[ii]/new_q[ii])
    return -div*dx

def sym_KLD(p, q, x, vep=1e-12):
    return .5*(continuous_KLD(p, q, x, vep) + continuous_KLD(q, p, x, vep))

def dis_sym_KLD(p, q, dx, vep=1e-12):
    return .5*(discrete_KLD(p, q, dx, vep) + discrete_KLD(q, p, dx, vep))

@vectorize
def epa_kernel(x):
    if np.abs(x) < 1:
        return .75*(1-x*x)
    else:
        return 0

@njit
def epa_density(x, x_vals, h):
    n = x_vals.size
    recip_h = 1/h
    running_sum = np.zeros(x.size)
    for xi in x_vals:
        running_sum += epa_kernel((x - xi)*recip_h)
    return recip_h*running_sum/n

@njit
def entropy_jit(p, dx, tol=1e-12):
    where = p > tol
    new_p = p[where]
    ent = 0
    for thing in new_p:
            ent += thing*np.log(thing)
    return -ent*dx

def get_divergences(data_set, supports, label='Enc divs'):

    divs = np.zeros(shape=(data_set.shape[0], data_set.shape[1]-1))
    
    for ii, (layer, support) in enumerate(zip(data_set, supports)):

        for jj in range(layer.shape[0]-1):
            divs[ii, jj] = dis_sym_KLD(layer[jj], layer[jj+1], support[1] - support[0])
        
        printProgressBar(ii, label, len(data_set))

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

def pdf_creator(weight_diffs, kde_kernel='epa', kde_bw='ISJ', label='Enc pdfs'):

    pdfs = [None]*weight_diffs.shape[1]
    supports = [None]*weight_diffs.shape[1]
    bands = [None]*weight_diffs.shape[1]
    simps_areas = [None]*weight_diffs.shape[1]
    entropies = [None]*weight_diffs.shape[1]


    for ii, layer in enumerate(weight_diffs.T):
        
        layer = np.array([ii for ii in layer])
        full_overshoot = .1*np.ptp(layer)
        full_support = np.linspace(layer.min()-full_overshoot, layer.max()+full_overshoot, 5001)
        dx = full_support[1] - full_support[0]
        layer_pdfs = [None]*layer.shape[0]
        layer_bandwidths = [None]*layer.shape[0]
        layer_simps_area = [None]*layer.shape[0]
        layer_entropies = [None]*layer.shape[0]


        for jj, data in enumerate(layer):
            
            estimator = FFTKDE(kernel=kde_kernel, bw=kde_bw).fit(data)
            layer_bandwidths[jj] = estimator.bw
            pdf = estimator.evaluate(full_support)
            # pdf = epa_density(full_support, data, estimator.bw) 
            #plt.plot(full_support, epa_density(full_support, data, estimator.bw))
            #plt.show()

            #simpson_area = integrate.quadrature(epa_density, *full_support[[0,1]], args=(data, estimator.bw),
            #                                    tol=1e-3)
            simpson_area = integrate.simpson(pdf, full_support)
            # print(simpson_area, end=" ")
            
            layer_simps_area[jj] = simpson_area
            layer_pdfs[jj] = pdf
            layer_entropies[jj] = entropy_jit(pdf, dx)

        
        pdfs[ii] = layer_pdfs
        supports[ii] = full_support
        bands[ii] = layer_bandwidths
        simps_areas[ii] = layer_simps_area
        entropies[ii] = layer_entropies

        printProgressBar(ii, label, weight_diffs.shape[1])

    return np.array(pdfs), np.array(supports), np.array(bands), np.array(simps_areas), np.array(entropies)

bw = "ISJ"

examples_folder = os.path.join(os.getcwd(), "examples")

models = [file for file in os.listdir(examples_folder) if os.path.splitext(file)[1] == '']

model_folders = [os.path.join(examples_folder, model) for model in models]

for model, model_name in zip(model_folders, models):
    print(f"Getting data from {model_name}.")
    trained_folder = os.path.join(model, "trained_models")


    for dim_run in os.listdir(trained_folder):
        print(dim_run)
        
        fold_components = dim_run.split('_')
        lat_dim = fold_components[-2]
        run_time = fold_components[-1]

        model_weight_folder = os.path.join(trained_folder, dim_run)
        weight_files = [os.path.join(model_weight_folder, "weights_by_epoch", file) for file in os.listdir(os.path.join(model_weight_folder, "weights_by_epoch")) if file.endswith('.h5')]
        model_weight_folder = os.path.join(model_weight_folder, f"bandwidth_{bw}")
        if not os.path.exists(model_weight_folder):
            os.mkdir(model_weight_folder)

        if len(weight_files) != 1000 or os.path.exists(os.path.join(model_weight_folder, 'divergence_results.csv')):
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
            printProgressBar(ii, "Weight Extraction", len(weight_files))

        del h5_file

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


        enc_pdfs, enc_supports, enc_bws, enc_simps, enc_entropies = pdf_creator(enc_weights_diff, kde_bw=bw)
        dec_pdfs, dec_supports, dec_bws, dec_simps, dec_entropies = pdf_creator(dec_weights_diff, kde_bw=bw, label='Dec pdfs')
        disties = np.array([enc_simps, dec_simps]).flatten()
        pkl.dump(disties, open(os.path.join(model_weight_folder, f"all_integrated_areas.pkl"),'wb'))

        full_weight_diff_dict = {**{lab:layer for lab, layer in zip(enc_labels, enc_weights_diff.T)}, 
                                 **{lab:layer for lab, layer in zip(dec_labels, dec_weights_diff.T)}}
        pkl.dump(full_weight_diff_dict, open(os.path.join(model_weight_folder, "all_weights_diff_array_dict.pkl"),'wb'))
        del dec_weights_diff, enc_weights_diff, full_weight_diff_dict

        pdf_dict = {**{lab:(layer, support) for lab, support, layer in zip(enc_labels, enc_supports, enc_pdfs)}, 
                    **{lab:(layer, support) for lab, support, layer in zip(dec_labels, dec_supports, dec_pdfs)}}
        pkl.dump(pdf_dict, open(os.path.join(model_weight_folder, "pdf_dict.pkl"),'wb'))

        enc_divergences = get_divergences(enc_pdfs, enc_supports)
        dec_divergences = get_divergences(dec_pdfs, dec_supports, label='Dec divs')        
        del enc_supports, dec_supports, enc_pdfs, dec_pdfs

        all_divergences = np.concatenate([enc_divergences, dec_divergences], axis=0).T
        all_entropies = np.concatenate([enc_entropies, dec_entropies], axis=0).T
        column_labels = enc_labels+dec_labels


        data_frame = pd.DataFrame(data=all_divergences, columns=column_labels, index=np.arange(all_divergences.shape[0])+1)
        data_frame.to_csv(os.path.join(model_weight_folder, "divergence_results.csv"))
        data_frame = pd.DataFrame(data=all_entropies, columns=column_labels, index=np.arange(all_entropies.shape[0])+1)
        data_frame.to_csv(os.path.join(model_weight_folder, "entropy_results.csv"))


        if isinstance(bw, str):
            all_bws = np.concatenate([enc_bws, dec_bws], axis=0).T
            data_frame = pd.DataFrame(data=all_bws, columns=column_labels, index=np.arange(all_bws.shape[0])+1)
            data_frame.to_csv(os.path.join(model_weight_folder, "bandwidth_results.csv"))
            del all_bws

        del enc_divergences, dec_divergences, all_divergences, all_entropies
        del enc_bws, dec_bws, enc_simps, dec_simps, data_frame, enc_entropies, dec_entropies
        print()