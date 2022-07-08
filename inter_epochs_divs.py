
import os
import sys
import h5py 
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from KDEpy import FFTKDE
import scipy.integrate as integrate
import datetime as dt
import matplotlib.pyplot as plt

def walk_h5(file_handle, space="", num_spaces=2):
    for key in file_handle.keys():
        print(space + key)
        try:
            walk_h5(file_handle[key], space=space + " "*num_spaces, num_spaces=num_spaces)
        except:
            pass

def discrete_kl_div(p, q, x, tol=1e-12):
    integrand = np.where(p > tol, p * np.log(p / q), 0)

    return integrate.simpson(integrand, x)
    # return integrate.trapezoid(integrand, x)

def sym_kl_divergence(p, q, x, vep=1e-12):
    return .5*(discrete_kl_div(p, q, x, vep) + discrete_kl_div(q, p, x, vep))

def div_between_pdfs(data_set1, data_set2, kde_kernel='epa'):
    discrete_points = 10001
    set_max = max((data_set1.max(), data_set2.max()))
    set_min = min((data_set1.min(), data_set2.min()))
    over_shoot = .01*(set_max - set_min)

    # _, bins1 = np.histogram(data_set1)
    # _, bins2 = np.histogram(data_set2)

    # bandwidth_1 = 0.2886751345948129*np.mean(bins1[1:] - bins1[:-1])
    # bandwidth_2 = 0.2886751345948129*np.mean(bins2[1:] - bins2[:-1])

    new_x = np.linspace(set_min-over_shoot, set_max+over_shoot, discrete_points)#.reshape(discrete_points, 1)
    data_set1 = data_set1.flatten()#.reshape(data_set1.size, 1)
    data_set2 = data_set2.flatten()#.reshape(data_set2.size, 1)


    estimator_1 = FFTKDE(kernel=kde_kernel, bw="ISJ").fit(data_set1)
    estimator_2 = FFTKDE(kernel=kde_kernel, bw="ISJ").fit(data_set2)

    bws = [estimator_1.bw, estimator_2.bw]
    
    pdf1 = estimator_1.evaluate(new_x)
    pdf2 = estimator_2.evaluate(new_x)

    # kde_1 = KernelDensity(kernel=kde_kernel, bandwidth=bandwidth_1).fit(data_set1)
    # kde_2 = KernelDensity(kernel=kde_kernel, bandwidth=bandwidth_2).fit(data_set2)

    # pdf1 = np.exp(kde_1.score_samples(new_x))
    # pdf2 = np.exp(kde_2.score_samples(new_x))

    simpson_area_1 = integrate.simpson(pdf1, new_x)#[:,0])
    simpson_area_2 = integrate.simpson(pdf2, new_x)#[:,0])
    
    pdf1 /= simpson_area_1
    pdf2 /= simpson_area_2

    # fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    # ax[0].hist(data_set1, bins='auto', density=True)
    # ax[0].hist(data_set2, bins='auto', density=True)

    # ax[1].plot(new_x, pdf1)
    # ax[1].plot(new_x, pdf2)

    # plt.show()
    # plt.close(fig)

    return sym_kl_divergence(pdf1, pdf2, new_x), bws

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
        
        fold_components = dim_run.split('_')
        lat_dim = fold_components[-2]
        run_time = fold_components[-1]

        model_weight_folder = os.path.join(trained_folder, dim_run)
        weight_files = [os.path.join(model_weight_folder, file) for file in os.listdir(model_weight_folder) if file.endswith('.h5')]

        if len(weight_files) == 1000 and not os.path.exists(os.path.join(model_weight_folder, 'divergence_results.csv')):
            h5_file_prev = h5py.File(weight_files[0], 'r')
        else:
            continue

        encoder_layers_prev = h5_file_prev['encoder']
        enc_labels = list(encoder_layers_prev.keys())
        enc_labels.insert(0, enc_labels.pop(-2))
        enc_weights_prev = [np.concatenate([encoder_layers_prev[key]['kernel:0'][:].flatten(), encoder_layers_prev[key]['bias:0'][:].flatten()]) for key in enc_labels]

        decoder_layers_prev = h5_file_prev['decoder']
        dec_labels = list(decoder_layers_prev.keys())
        dec_labels.insert(0, dec_labels.pop(-2))
        dec_weights_prev = [np.concatenate([decoder_layers_prev[key]['kernel:0'][:].flatten(), decoder_layers_prev[key]['bias:0'][:].flatten()]) for key in dec_labels]


        enc_weight_diffs = [None]*len(weight_files[1:]) 
        dec_weight_diffs = [None]*len(weight_files[1:])


        for ii, weight_file in enumerate(weight_files[1:]):
            
            h5_file_next = h5py.File(weight_file, 'r')


            encoder_layers_next = h5_file_next['encoder']
            enc_weights_next = [np.concatenate([encoder_layers_next[key]['kernel:0'][:].flatten(), encoder_layers_next[key]['bias:0'][:].flatten()]) for key in enc_labels]

            decoder_layers_next = h5_file_next['decoder']
            dec_weights_next = [np.concatenate([decoder_layers_next[key]['kernel:0'][:].flatten(), decoder_layers_next[key]['bias:0'][:].flatten()]) for key in dec_labels]


            enc_weight_diffs[ii] = [coming - prev for coming, prev in zip(enc_weights_next, enc_weights_prev)]
            dec_weight_diffs[ii] = [coming - prev for coming, prev in zip(dec_weights_next, dec_weights_prev)]
            

            enc_weights_prev = enc_weights_next
            dec_weights_prev = dec_weights_next

        
        h5_file_prev.close()
        h5_file_next.close()
        del h5_file_prev
        del h5_file_next

        enc_weight_divs = [None]*len(enc_weight_diffs[1:])
        dec_weight_divs = [None]*len(enc_weight_diffs[1:])

        enc_bws = [None]*len(enc_weight_diffs)
        dec_bws = [None]*len(dec_weight_diffs)

        for ii in range(len(enc_weight_diffs[1:])):

            enc_weight_diffs_prev = enc_weight_diffs[ii]
            dec_weight_diffs_prev = dec_weight_diffs[ii]

            enc_weight_diffs_next = enc_weight_diffs[ii+1]
            dec_weight_diffs_next = dec_weight_diffs[ii+1]

            enc_temp, dec_temp = [None]*len(enc_weight_diffs_next), [None]*len(dec_weight_diffs_next)
            enc_bw_temp, dec_bw_temp = [None]*len(enc_weight_diffs_next), [None]*len(dec_weight_diffs_next)

            for qq, (data1, data2, data3, data4) in enumerate(zip(enc_weight_diffs_next, enc_weight_diffs_prev, dec_weight_diffs_next, dec_weight_diffs_prev)):
                enc_div, enc_bw = div_between_pdfs(data1, data2)
                dec_div, dec_bw = div_between_pdfs(data3, data4)

                enc_temp[qq], dec_temp[qq] = enc_div, dec_div
                enc_bw_temp[qq], dec_bw_temp[qq] = enc_bw, dec_bw

            enc_bw_temp = np.transpose(enc_bw_temp)
            dec_bw_temp = np.transpose(dec_bw_temp)


            enc_bws[ii:ii+2] = enc_bw_temp
            dec_bws[ii:ii+2] = dec_bw_temp

            enc_weight_divs[ii] = enc_temp#[div_between_pdfs(data1, data2) for data1, data2 in zip(enc_weight_diffs_next, enc_weight_diffs_prev)]
            dec_weight_divs[ii] = dec_temp#[div_between_pdfs(data3, data4) for data3, data4 in zip(dec_weight_diffs_next, dec_weight_diffs_prev)]

            printProgressBar(ii, f"Model: {model_name}, Latent Dim: {lat_dim}, Run Time: {run_time}", len(enc_weight_diffs[1:]))

        all_divs = np.concatenate([enc_weight_divs, dec_weight_divs], axis=1)
        all_bws = np.concatenate([enc_bws, dec_bws], axis=1)
        
        column_labels = enc_labels + dec_labels

        data_frame = pd.DataFrame(data=all_divs, columns=column_labels, index=np.arange(all_divs.shape[0])+1)
        data_frame.to_csv(os.path.join(model_weight_folder, "divergence_results.csv"))

        data_frame = pd.DataFrame(data=all_bws, columns=column_labels, index=np.arange(all_bws.shape[0])+1)
        data_frame.to_csv(os.path.join(model_weight_folder, "bandwidth_results.csv"))
