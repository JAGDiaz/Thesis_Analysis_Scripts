
import os
import sys
import h5py 
import numpy as np
import pandas as pd
from KDEpy import TreeKDE, NaiveKDE
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pickle as pkl
from numba import vectorize
import warnings

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

def density_maker(x_vals, h, freqs, kernel=epa_kernel):
    n = x_vals.size
    def density(x):
        X, Xi = np.meshgrid(x, x_vals)
        values = freqs.reshape(freqs.size,1)*kernel((X - Xi)/h)
        
        return np.sum(values, axis=0)/(n*h)
    return density

def get_diveregences(data_set, supports, label='Enc divs'):

    divs = np.zeros(shape=(data_set.shape[0], data_set.shape[1]-1))
    
    for ii, (layer, support) in enumerate(zip(data_set, supports)):

        for jj in range(layer.shape[0]-1):
            divs[ii, jj] = sym_KLD(layer[jj], layer[jj+1], support)
        
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


    for ii, layer in enumerate(weight_diffs.T):
        
        layer = np.array([ii for ii in layer])
        full_overshoot = .1*np.ptp(layer)
        full_support = np.linspace(layer.min()-full_overshoot, layer.max()+full_overshoot, 5001)
        layer_pdfs = [None]*layer.shape[0]
        layer_bandwidths = [None]*layer.shape[0]
        layer_simps_area = [None]*layer.shape[0]

        for jj, data in enumerate(layer):
            
            # layer_overshoot = .01*np.ptp(data)
            # layer_support = np.linspace(data.min()-layer_overshoot, data.max()+layer_overshoot, full_support.size)

            estimator = NaiveKDE(kernel=kde_kernel, bw=kde_bw).fit(data)
            layer_bandwidths[jj] = estimator.bw
            #bins_number = int(np.ptp(data)/estimator.bw)+1
            #hist, bins = np.histogram(data, bins_number)
            #bins = .5*(bins[:-1] + bins[1:])
            pdf = estimator.evaluate(full_support)
            #layer_func = density_maker(bins, estimator.bw, hist)
            #pdf = layer_func(full_support)
            # simpson_area = integrate.trapezoid(pdf, full_support)
            simpson_area = integrate.quadrature(estimator.evaluate, *full_support[[0,-1]], maxiter=1000, tol=1e-3)
            # print(simpson_area,end=" ")

            
            # print(simpson_area, end=" ")
            # layer_simps_area[jj] = integrate.quadrature(layer_func, *full_support[[0,-1]])[0]
            
            layer_simps_area[jj] = simpson_area[0]
            layer_pdfs[jj] = pdf
        
        pdfs[ii] = layer_pdfs
        supports[ii] = full_support
        bands[ii] = layer_bandwidths
        simps_areas[ii] = layer_simps_area

        printProgressBar(ii, label, weight_diffs.shape[1])

    return np.array(pdfs), np.array(supports), np.array(bands), np.array(simps_areas)

examples_folder = os.path.join(os.getcwd(), "examples")

models = [file for file in os.listdir(examples_folder) if os.path.splitext(file)[-1] == '']

model_folders = [os.path.join(examples_folder, model) for model in models]

# fug, uxes = plt.subplots(1,len(model_folders), figsize=(5*len(model_folders),5), sharey='row')

for model, model_name in zip(model_folders, models):
    print(f"Generating data for {model_name}.")
    trained_folder = os.path.join(model, "trained_models")

    disties = []

    for dim_run in os.listdir(trained_folder):
        print(dim_run)
        
        fold_components = dim_run.split('_')
        lat_dim = fold_components[-2]
        run_time = fold_components[-1]

        model_weight_folder = os.path.join(trained_folder, dim_run)
        weight_files = [os.path.join(model_weight_folder, "weights_by_epoch", file) for file in os.listdir(model_weight_folder) if file.endswith('.h5')]

        if len(weight_files) != 1000:# or os.path.exists(os.path.join(model_weight_folder, 'divergence_results.csv')):
            continue

        weights = [None]*len(weight_files)

        for ii, weight_file in enumerate(weight_files):

            h5_file = h5py.File(weight_file, 'r')
            encoder_layers = h5_file['encoder']
            enc_labels = list(encoder_layers.keys())
            enc_labels.insert(0, enc_labels.pop(-2))

            decoder_layers = h5_file['decoder']
            dec_labels = list(decoder_layers.keys())
            dec_labels.insert(0, dec_labels.pop(-2))

            full_net = np.concatenate([np.concatenate([encoder_layers[key]['kernel:0'][:].flatten(), encoder_layers[key]['bias:0'][:].flatten()]) for key in enc_labels]+
                                      [np.concatenate([decoder_layers[key]['kernel:0'][:].flatten(), decoder_layers[key]['bias:0'][:].flatten()]) for key in dec_labels])

            h5_file.close()

            weights[ii] = full_net 
            printProgressBar(ii, "Weight Extraction", len(weight_files))

        del h5_file

        weights = np.array(weights)

        diff_weights = np.diff(weights, axis=0)

        del weights

        for diff in diff_weights:

            n, _, _ = plt.hist(diff, bins='auto', density=True)
            plt.xlim(diff.min(), diff.max())
            plt.ylim(0, 1.1*np.max(n))
            plt.show()







        
