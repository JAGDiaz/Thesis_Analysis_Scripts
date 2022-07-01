
import os
import h5py 
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import scipy.integrate as integrate
import scipy.stats as stats
import scipy.optimize as optimize
import datetime as dt
import matplotlib.pyplot as plt


def discrete_kl_div(p,q, tol=1e-12):
    return np.sum(np.where(p > tol, p * np.log(p / q), 0))

def sym_kl_divergence(p, q, vep=1e-12):
    return .5*(discrete_kl_div(p, q, vep) + discrete_kl_div(q, p, vep))

def div_between_pdfs(data_set1, data_set2, kde_kernel='exponential', kde_bandwidth=.4,
                     discrete_points=1001):
    set_max = max((data_set1.max(), data_set2.max()))
    set_min = min((data_set1.min(), data_set2.min()))

    new_x = np.linspace(set_min, set_max, discrete_points)[:, np.newaxis]

    kde_1 = KernelDensity(kernel=kde_kernel, bandwidth=kde_bandwidth).fit(data_set1.flatten()[:, np.newaxis])
    kde_2 = KernelDensity(kernel=kde_kernel, bandwidth=kde_bandwidth).fit(data_set2.flatten()[:, np.newaxis])

    pdf1 = np.exp(kde_1.score_samples(new_x))
    pdf2 = np.exp(kde_2.score_samples(new_x))

    simpson_area_1 = integrate.simpson(pdf1, new_x[:,0])
    simpson_area_2 = integrate.simpson(pdf2, new_x[:,0])
    
    
    pdf1 /= simpson_area_1
    pdf2 /= simpson_area_2

    return sym_kl_divergence(pdf1, pdf2)

def get_euclidean(first_set, second_set):
    
    metrics_1 = [None]*len(first_set)
    metrics_2 = [None]*len(first_set)
    metrics_diff = [None]*len(first_set)
    for ii, (set1, set2) in enumerate(zip(first_set, second_set)):
        metrics_diff[ii] = np.linalg.norm(set1 - set2, ord=2)
        metrics_1[ii] = np.linalg.norm(set1 - set2, ord=2)/np.linalg.norm(set1, ord=2)
        metrics_1[ii] = np.linalg.norm(set1 - set2, ord=2)/np.linalg.norm(set1, ord=2)
    
    return metrics_1, metrics_2

def get_div_euclid_and_moments(sets1, sets2, moments=False):

    sets_length = len(sets1)

    metrics_1 = [None]*sets_length
    metrics_2 = [None]*sets_length
    metrics_diff = [None]*sets_length
    div = [None]*sets_length

    if moments:
        kurtosis = [None]*sets_length
        mean = [None]*sets_length
        skewness = [None]*sets_length
        variance = [None]*sets_length

        for jj, (set1, set2) in enumerate(zip(sets1, sets2)):
            div[jj] = div_between_pdfs(set1, set2)
            metrics_1[jj] = np.linalg.norm(set1, ord=2)
            metrics_2[jj] = np.linalg.norm(set2, ord=2)
            metrics_diff[jj] = np.linalg.norm(set1 - set2, ord=2)
            
            kurtosis[jj] = stats.kurtosis([set1, set2], axis=None)
            mean[jj] = np.mean([set1, set2])

            skewness[jj] = stats.skew([set1, set2], axis=None)
            variance[jj] = np.var([set1, set2])

        return [div, metrics_1, metrics_2, metrics_diff, mean, variance, skewness, kurtosis]

    else:
        for jj, (set1, set2) in enumerate(zip(sets1, sets2)):
            div[jj] = div_between_pdfs(set1, set2)
            metrics_1[jj] = np.linalg.norm(set1, ord=2)
            metrics_2[jj] = np.linalg.norm(set2, ord=2)
            metrics_diff[jj] = np.linalg.norm(set1 - set2, ord=2)
        
        return [div, metrics_1, metrics_2, metrics_diff]

def h5s_to_dict(h5_list):
    the_dict = {}
    for ii, h5 in enumerate(h5_list):
        the_dict[ii+1] = {}
        with h5py.File(h5, 'r') as f:
            encoder_layers = f['encoder']
            for key in encoder_layers.keys():
                for sub_key in encoder_layers[key]:
                    the_dict[ii+1][key + ":" + sub_key] = encoder_layers[key][sub_key][:]

            decoder_layers = f['decoder']
            for key in decoder_layers.keys():
                for sub_key in decoder_layers[key]:
                    the_dict[ii+1][key + ":" + sub_key] = decoder_layers[key][sub_key][:]

    return the_dict

def walk_h5(file_handle, space="", num_spaces=2):
    for key in file_handle.keys():
        print(space + key)
        try:
            walk_h5(file_handle[key], space=space + " "*num_spaces, num_spaces=num_spaces)
        except:
            pass

moments = True
make_plots = False

#latdim = 2
current_model = 'van_der_pol'
run_time = f"{current_model}_7_2021-11-11-2239"

weight_folder = f"C:\\Users\\josep\\Documents\\School Work\\Research\\Curtis\\Machine_Learning\\my_work\\DLDMD-newest\\examples\\{current_model}\\trained_models\\{run_time}"


paths = [os.path.join(weight_folder, file) for file in os.listdir(weight_folder) if file.endswith('.h5')]



beginning_time = dt.datetime.now()
weight_comp_path = f".\weight_comp\{current_model}-single\{run_time}"
if not os.path.exists(weight_comp_path):
    os.makedirs(weight_comp_path)


enc_weight_divs_epoch = [None]*(len(paths) - 1)
enc_biases_divs_epoch = [None]*(len(paths) - 1)
dec_weight_divs_epoch = [None]*(len(paths) - 1)
dec_biases_divs_epoch = [None]*(len(paths) - 1)

enc_1_weight_metric_epoch = [None]*(len(paths) - 1)
enc_1_biases_metric_epoch = [None]*(len(paths) - 1)
dec_1_weight_metric_epoch = [None]*(len(paths) - 1)
dec_1_biases_metric_epoch = [None]*(len(paths) - 1)

enc_2_weight_metric_epoch = [None]*(len(paths) - 1)
enc_2_biases_metric_epoch = [None]*(len(paths) - 1)
dec_2_weight_metric_epoch = [None]*(len(paths) - 1)
dec_2_biases_metric_epoch = [None]*(len(paths) - 1)

enc_diff_weight_metric_epoch = [None]*(len(paths) - 1)
enc_diff_biases_metric_epoch = [None]*(len(paths) - 1)
dec_diff_weight_metric_epoch = [None]*(len(paths) - 1)
dec_diff_biases_metric_epoch = [None]*(len(paths) - 1)

if moments:
    enc_weight_mean_epoch = [None]*(len(paths) - 1)
    enc_biases_mean_epoch = [None]*(len(paths) - 1)
    dec_weight_mean_epoch = [None]*(len(paths) - 1)
    dec_biases_mean_epoch = [None]*(len(paths) - 1)

    enc_weight_vari_epoch = [None]*(len(paths) - 1)
    enc_biases_vari_epoch = [None]*(len(paths) - 1)
    dec_weight_vari_epoch = [None]*(len(paths) - 1)
    dec_biases_vari_epoch = [None]*(len(paths) - 1)

    enc_weight_skew_epoch = [None]*(len(paths) - 1)
    enc_biases_skew_epoch = [None]*(len(paths) - 1)
    dec_weight_skew_epoch = [None]*(len(paths) - 1)
    dec_biases_skew_epoch = [None]*(len(paths) - 1)

    enc_weight_kurt_epoch = [None]*(len(paths) - 1)
    enc_biases_kurt_epoch = [None]*(len(paths) - 1)
    dec_weight_kurt_epoch = [None]*(len(paths) - 1)
    dec_biases_kurt_epoch = [None]*(len(paths) - 1)


h5_file_1 = h5py.File(paths[0], 'r')

encoder_layers_1 = h5_file_1['encoder']
enc_labels = list(encoder_layers_1.keys())
enc_weights_1 = [encoder_layers_1[key]['kernel:0'][:].flatten() for key in enc_labels]
enc_biases_1 = [encoder_layers_1[key]['bias:0'][:].flatten() for key in enc_labels]

#enc_weights_1[0] = np.concatenate([enc_weights_1[0], enc_biases_1.pop(0)])
enc_weights_1[-1] = np.concatenate([enc_weights_1[-1], enc_biases_1.pop(-1)])


decoder_layers_1 = h5_file_1['decoder']
dec_labels = list(decoder_layers_1.keys())
dec_weights_1 = [decoder_layers_1[key]['kernel:0'][:].flatten() for key in dec_labels]
dec_biases_1 = [decoder_layers_1[key]['bias:0'][:].flatten() for key in dec_labels]

#dec_weights_1[0] = np.concatenate([dec_weights_1[0], dec_biases_1.pop(0)])
dec_weights_1[-1] = np.concatenate([dec_weights_1[-1], dec_biases_1.pop(-1)])

for ii in range(1, len(paths)):
    h5_file_2 = h5py.File(paths[ii], 'r')


    encoder_layers_2 = h5_file_2['encoder']
    enc_weights_2 = [encoder_layers_2[key]['kernel:0'][:].flatten() for key in enc_labels]
    enc_biases_2 = [encoder_layers_2[key]['bias:0'][:].flatten() for key in enc_labels]

    enc_weights_2[-1] = np.concatenate([enc_weights_2[-1], enc_biases_2.pop(-1)])


    decoder_layers_2 = h5_file_2['decoder']
    dec_weights_2 = [decoder_layers_2[key]['kernel:0'][:].flatten() for key in dec_labels]
    dec_biases_2 = [decoder_layers_2[key]['bias:0'][:].flatten() for key in dec_labels]

    dec_weights_2[-1] = np.concatenate([dec_weights_2[-1], dec_biases_2.pop(-1)])


    # Compute the divergence, euclidean distance, and moments
    enc_weight = get_div_euclid_and_moments(enc_weights_1, enc_weights_2, moments)
    enc_biases = get_div_euclid_and_moments(enc_biases_1, enc_biases_2, moments)
    dec_weight = get_div_euclid_and_moments(dec_weights_1, dec_weights_2, moments)
    dec_biases = get_div_euclid_and_moments(dec_biases_1, dec_biases_2, moments)

    # Pull out the metrics we want.
    enc_weight_div, enc_1_weight_metric, enc_2_weight_metric, enc_diff_weight_metric = enc_weight[:4]
    enc_biases_div, enc_1_biases_metric, enc_2_biases_metric, enc_diff_biases_metric = enc_biases[:4]
    dec_weight_div, dec_1_weight_metric, dec_2_weight_metric, dec_diff_weight_metric = dec_weight[:4]
    dec_biases_div, dec_1_biases_metric, dec_2_biases_metric, dec_diff_biases_metric = dec_biases[:4]


    # Store the euclidean distances in these lists...
    enc_1_weight_metric_epoch[ii - 1] = enc_1_weight_metric
    enc_1_biases_metric_epoch[ii - 1] = enc_1_biases_metric
    dec_1_weight_metric_epoch[ii - 1] = dec_1_weight_metric
    dec_1_biases_metric_epoch[ii - 1] = dec_1_biases_metric

    enc_2_weight_metric_epoch[ii - 1] = enc_2_weight_metric
    enc_2_biases_metric_epoch[ii - 1] = enc_2_biases_metric
    dec_2_weight_metric_epoch[ii - 1] = dec_2_weight_metric
    dec_2_biases_metric_epoch[ii - 1] = dec_2_biases_metric

    enc_diff_weight_metric_epoch[ii - 1] = enc_diff_weight_metric
    enc_diff_biases_metric_epoch[ii - 1] = enc_diff_biases_metric
    dec_diff_weight_metric_epoch[ii - 1] = dec_diff_weight_metric
    dec_diff_biases_metric_epoch[ii - 1] = dec_diff_biases_metric

    # ... And the divergences in these lists.
    enc_weight_divs_epoch[ii - 1] = enc_weight_div
    enc_biases_divs_epoch[ii - 1] = enc_biases_div
    dec_weight_divs_epoch[ii - 1] = dec_weight_div
    dec_biases_divs_epoch[ii - 1] = dec_biases_div

    # If we computed the moments, this is when we pull them and store them:
    if moments:
        enc_weight_mean, enc_weight_varr, enc_weight_skew, enc_weight_kurt = enc_weight[4:]
        enc_biases_mean, enc_biases_varr, enc_biases_skew, enc_biases_kurt = enc_biases[4:]
        dec_weight_mean, dec_weight_varr, dec_weight_skew, dec_weight_kurt = dec_weight[4:]
        dec_biases_mean, dec_biases_varr, dec_biases_skew, dec_biases_kurt = dec_biases[4:]


        enc_weight_mean_epoch[ii - 1] = enc_weight_mean
        enc_biases_mean_epoch[ii - 1] = enc_biases_mean
        dec_weight_mean_epoch[ii - 1] = dec_weight_mean
        dec_biases_mean_epoch[ii - 1] = dec_biases_mean

        enc_weight_vari_epoch[ii - 1] = enc_weight_varr
        enc_biases_vari_epoch[ii - 1] = enc_biases_varr
        dec_weight_vari_epoch[ii - 1] = dec_weight_varr
        dec_biases_vari_epoch[ii - 1] = dec_biases_varr

        enc_weight_skew_epoch[ii - 1] = enc_weight_skew
        enc_biases_skew_epoch[ii - 1] = enc_biases_skew
        dec_weight_skew_epoch[ii - 1] = dec_weight_skew
        dec_biases_skew_epoch[ii - 1] = dec_biases_skew

        enc_weight_kurt_epoch[ii - 1] = enc_weight_kurt
        enc_biases_kurt_epoch[ii - 1] = enc_biases_kurt
        dec_weight_kurt_epoch[ii - 1] = dec_weight_kurt
        dec_biases_kurt_epoch[ii - 1] = dec_biases_kurt


    enc_weights_1 = enc_weights_2
    enc_biases_1 = enc_biases_2

    dec_weights_1 = dec_weights_2
    dec_biases_1 = dec_biases_2

    h5_file_1.close()
    h5_file_1 = h5_file_2
    print(f"Inter-Epoch {ii} Done.")
    if not ii % 100:
        print(f"Running time so far: {dt.datetime.now() - beginning_time}.")




h5_file_1.close()
h5_file_2.close()
del h5_file_1
del h5_file_2

enc_1_weight_metric_epoch = np.array(enc_1_weight_metric_epoch)
enc_1_biases_metric_epoch = np.array(enc_1_biases_metric_epoch)
dec_1_weight_metric_epoch = np.array(dec_1_weight_metric_epoch)
dec_1_biases_metric_epoch = np.array(dec_1_biases_metric_epoch)

enc_2_weight_metric_epoch = np.array(enc_2_weight_metric_epoch)
enc_2_biases_metric_epoch = np.array(enc_2_biases_metric_epoch)
dec_2_weight_metric_epoch = np.array(dec_2_weight_metric_epoch)
dec_2_biases_metric_epoch = np.array(dec_2_biases_metric_epoch)

enc_diff_weight_metric_epoch = np.array(enc_diff_weight_metric_epoch)
enc_diff_biases_metric_epoch = np.array(enc_diff_biases_metric_epoch)
dec_diff_weight_metric_epoch = np.array(dec_diff_weight_metric_epoch)
dec_diff_biases_metric_epoch = np.array(dec_diff_biases_metric_epoch)

enc_weight_divs_epoch = np.array(enc_weight_divs_epoch)
enc_biases_divs_epoch = np.array(enc_biases_divs_epoch)
dec_weight_divs_epoch = np.array(dec_weight_divs_epoch)
dec_biases_divs_epoch = np.array(dec_biases_divs_epoch)

epochs = np.arange(enc_weight_divs_epoch.shape[0]) + 1

# Save the generated data into csvs.
pd.DataFrame(columns=enc_labels, data=enc_weight_divs_epoch, index=epochs).to_csv(f"{weight_comp_path}\\encoder_weights_divergences.csv")
pd.DataFrame(columns=enc_labels[:-1], data=enc_biases_divs_epoch, index=epochs).to_csv(f"{weight_comp_path}\\encoder_biases_divergences.csv")
pd.DataFrame(columns=dec_labels, data=dec_weight_divs_epoch, index=epochs).to_csv(f"{weight_comp_path}\\decoder_weights_divergences.csv")
pd.DataFrame(columns=dec_labels[:-1], data=dec_biases_divs_epoch, index=epochs).to_csv(f"{weight_comp_path}\\decoder_biases_divergences.csv")

pd.DataFrame(columns=enc_labels, data=enc_1_weight_metric_epoch, index=epochs).to_csv(f"{weight_comp_path}\\encoder_1_weights_euclids.csv")
pd.DataFrame(columns=enc_labels[:-1], data=enc_1_biases_metric_epoch, index=epochs).to_csv(f"{weight_comp_path}\\encoder_1_biases_euclids.csv")
pd.DataFrame(columns=dec_labels, data=dec_1_weight_metric_epoch, index=epochs).to_csv(f"{weight_comp_path}\\decoder_1_weights_euclids.csv")
pd.DataFrame(columns=dec_labels[:-1], data=dec_1_biases_metric_epoch, index=epochs).to_csv(f"{weight_comp_path}\\decoder_1_biases_euclids.csv")

pd.DataFrame(columns=enc_labels, data=enc_2_weight_metric_epoch, index=epochs).to_csv(f"{weight_comp_path}\\encoder_2_weights_euclids.csv")
pd.DataFrame(columns=enc_labels[:-1], data=enc_2_biases_metric_epoch, index=epochs).to_csv(f"{weight_comp_path}\\encoder_2_biases_euclids.csv")
pd.DataFrame(columns=dec_labels, data=dec_2_weight_metric_epoch, index=epochs).to_csv(f"{weight_comp_path}\\decoder_2_weights_euclids.csv")
pd.DataFrame(columns=dec_labels[:-1], data=dec_2_biases_metric_epoch, index=epochs).to_csv(f"{weight_comp_path}\\decoder_2_biases_euclids.csv")

pd.DataFrame(columns=enc_labels, data=enc_diff_weight_metric_epoch, index=epochs).to_csv(f"{weight_comp_path}\\encoder_diff_weights_euclids.csv")
pd.DataFrame(columns=enc_labels[:-1], data=enc_diff_biases_metric_epoch, index=epochs).to_csv(f"{weight_comp_path}\\encoder_diff_biases_euclids.csv")
pd.DataFrame(columns=dec_labels, data=dec_diff_weight_metric_epoch, index=epochs).to_csv(f"{weight_comp_path}\\decoder_diff_weights_euclids.csv")
pd.DataFrame(columns=dec_labels[:-1], data=dec_diff_biases_metric_epoch, index=epochs).to_csv(f"{weight_comp_path}\\decoder_diff_biases_euclids.csv")


if moments:
    enc_weight_mean_epoch = np.array(enc_weight_mean_epoch)
    enc_biases_mean_epoch = np.array(enc_biases_mean_epoch)
    dec_weight_mean_epoch = np.array(dec_weight_mean_epoch)
    dec_biases_mean_epoch = np.array(dec_biases_mean_epoch)

    pd.DataFrame(columns=enc_labels, data=enc_weight_mean_epoch, index=epochs).to_csv(f"{weight_comp_path}\\encoder_weights_means.csv")
    pd.DataFrame(columns=enc_labels[:-1], data=enc_biases_mean_epoch, index=epochs).to_csv(f"{weight_comp_path}\\encoder_biases_means.csv")
    pd.DataFrame(columns=dec_labels, data=dec_weight_mean_epoch, index=epochs).to_csv(f"{weight_comp_path}\\decoder_weights_means.csv")
    pd.DataFrame(columns=dec_labels[:-1], data=dec_biases_mean_epoch, index=epochs).to_csv(f"{weight_comp_path}\\decoder_biases_means.csv")


    enc_weight_vari_epoch = np.array(enc_weight_vari_epoch)
    enc_biases_vari_epoch = np.array(enc_biases_vari_epoch)
    dec_weight_vari_epoch = np.array(dec_weight_vari_epoch)
    dec_biases_vari_epoch = np.array(dec_biases_vari_epoch)

    pd.DataFrame(columns=enc_labels, data=enc_weight_vari_epoch, index=epochs).to_csv(f"{weight_comp_path}\\encoder_weights_varies.csv")
    pd.DataFrame(columns=enc_labels[:-1], data=enc_biases_vari_epoch, index=epochs).to_csv(f"{weight_comp_path}\\encoder_biases_varies.csv")
    pd.DataFrame(columns=dec_labels, data=dec_weight_vari_epoch, index=epochs).to_csv(f"{weight_comp_path}\\decoder_weights_varies.csv")
    pd.DataFrame(columns=dec_labels[:-1], data=dec_biases_vari_epoch, index=epochs).to_csv(f"{weight_comp_path}\\decoder_biases_varies.csv")


    enc_weight_skew_epoch = np.array(enc_weight_skew_epoch)
    enc_biases_skew_epoch = np.array(enc_biases_skew_epoch)
    dec_weight_skew_epoch = np.array(dec_weight_skew_epoch)
    dec_biases_skew_epoch = np.array(dec_biases_skew_epoch)

    pd.DataFrame(columns=enc_labels, data=enc_weight_skew_epoch, index=epochs).to_csv(f"{weight_comp_path}\\encoder_weights_skews.csv")
    pd.DataFrame(columns=enc_labels[:-1], data=enc_biases_skew_epoch, index=epochs).to_csv(f"{weight_comp_path}\\encoder_biases_skews.csv")
    pd.DataFrame(columns=dec_labels, data=dec_weight_skew_epoch, index=epochs).to_csv(f"{weight_comp_path}\\decoder_weights_skews.csv")
    pd.DataFrame(columns=dec_labels[:-1], data=dec_biases_skew_epoch, index=epochs).to_csv(f"{weight_comp_path}\\decoder_biases_skews.csv")


    enc_weight_kurt_epoch = np.array(enc_weight_kurt_epoch)
    enc_biases_kurt_epoch = np.array(enc_biases_kurt_epoch)
    dec_weight_kurt_epoch = np.array(dec_weight_kurt_epoch)
    dec_biases_kurt_epoch = np.array(dec_biases_kurt_epoch)

    pd.DataFrame(columns=enc_labels, data=enc_weight_kurt_epoch, index=epochs).to_csv(f"{weight_comp_path}\\encoder_weights_kurts.csv")
    pd.DataFrame(columns=enc_labels[:-1], data=enc_biases_kurt_epoch, index=epochs).to_csv(f"{weight_comp_path}\\encoder_biases_kurts.csv")
    pd.DataFrame(columns=dec_labels, data=dec_weight_kurt_epoch, index=epochs).to_csv(f"{weight_comp_path}\\decoder_weights_kurts.csv")
    pd.DataFrame(columns=dec_labels[:-1], data=dec_biases_kurt_epoch, index=epochs).to_csv(f"{weight_comp_path}\\decoder_biases_kurts.csv")



if make_plots:
    os.makedirs(weight_comp_path+"\\log_plots")

    # Divergence plots - weights #
    fig, ax = plt.subplots(figsize=(10, 7))

    for datum, label in zip(enc_weight_divs_epoch.T[:-1], enc_labels[:-1]):
        ax.scatter(epochs, datum, alpha=.75, label=label)

    for datum, label in zip(dec_weight_divs_epoch.T[:-1], dec_labels[:-1]):
        ax.scatter(epochs, datum, alpha=.75, label=label)

    ax.set_xlabel("Inter-Epoch $i$", size=20)
    ax.set_ylabel("$KL(p_{i+1} || p_i)$", size=20)

    ax.tick_params(axis='both', length=10, labelsize=15)

    ax.legend(loc='best', fontsize=10, ncol=4)
    ax.grid(True)
    ax.set_title(f"{current_model.capitalize()}-single; {epochs[-1] + 1} Epochs; Weights; {run_time}.".replace("_", " "))

    fig.savefig(f"{weight_comp_path}\\divergence_plot.jpg", dpi=400)

    ax.set_yscale('log')
    ax.set_ylabel("$\\log_{10 }KL(p_{i+1} || p_i)$", size=20)
    fig.savefig(f"{weight_comp_path}\\log_plots\\divergence_plot_log.jpg", dpi=400)
    plt.close(fig)


    # Divergence plots - biases #
    fig, ax = plt.subplots(figsize=(10, 7))

    for datum, label in zip(enc_biases_divs_epoch.T, enc_labels[:-1]):
        ax.scatter(epochs, datum, alpha=.75, label=label)

    for datum, label in zip(dec_biases_divs_epoch.T, dec_labels[:-1]):
        ax.scatter(epochs, datum, alpha=.75, label=label)

    ax.set_xlabel("Inter-Epoch $i$", size=20)
    ax.set_ylabel("$KL(p_{i+1} || p_i)$", size=20)

    ax.tick_params(axis='both', length=10, labelsize=15)

    ax.legend(loc='best', fontsize=10, ncol=4)
    ax.grid(True)
    ax.set_title(f"{current_model.capitalize()}-single; {epochs[-1]+1} Epochs; Biases; {run_time}.".replace("_", " "))

    fig.savefig(f"{weight_comp_path}\\divergence_biases_plot.jpg", dpi=400)

    ax.set_yscale('log')
    ax.set_ylabel("$\\log_{10 }KL(p_{i+1} || p_i)$", size=20)
    fig.savefig(f"{weight_comp_path}\\log_plots\\divergence_biases_plot_log.jpg", dpi=400)

    plt.close(fig)

    # Euclidean plots - weights #
    fig, ax = plt.subplots(figsize=(10, 7))

    for datum, label in zip(enc_1_weight_metric_epoch.T[:-1], enc_labels[:-1]):
        ax.scatter(epochs, datum, alpha=.75, label=label)

    for datum, label in zip(dec_1_weight_metric_epoch.T[:-1], dec_labels[:-1]):
        ax.scatter(epochs, datum, alpha=.75, label=label)

    ax.set_xlabel("Inter-Epoch $i$", size=20)
    ax.set_ylabel("$||W_{i + 1} - W_i||_2/||W_i||_2$", size=20)

    ax.tick_params(axis='both', length=10, labelsize=15)

    ax.legend(loc='best', fontsize=10, ncol=4)
    ax.grid(True)
    ax.set_title(f"{current_model.capitalize()}-single; {epochs[-1] + 1} Epochs; Weights; {run_time}.".replace("_", " "))

    fig.savefig(f"{weight_comp_path}\\euclidean_plot.jpg", dpi=400)

    ax.set_yscale('log')
    ax.set_ylabel("$\\log_{10 }||W_{i + 1} - W_i||_2/||W_i||_2$", size=20)
    fig.savefig(f"{weight_comp_path}\\log_plots\\euclidean_plot_log.jpg", dpi=400)
    plt.close(fig)

    # Euclidean Enc_out, Dec_out 
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(epochs, enc_1_weight_metric_epoch[:, -1], alpha=.75, label=enc_labels[-1])
    ax.scatter(epochs, dec_1_weight_metric_epoch[:, -1], alpha=.75, label=dec_labels[-1])

    ax.set_xlabel("Inter-Epoch $i$", size=20)
    ax.set_ylabel("$||W_{i + 1} - W_i||_2/||W_i||_2$", size=20)

    ax.tick_params(axis='both', length=10, labelsize=15)

    ax.legend(loc='best', fontsize=10, ncol=4)
    ax.grid(True)
    ax.set_title(f"{current_model.capitalize()}-single; {epochs[-1] + 1} Epochs; Weights; {run_time}.".replace("_", " "))

    fig.savefig(f"{weight_comp_path}\\euclidean_plot_outs.jpg", dpi=400)


    # Euclidean plots - biases #
    fig, ax = plt.subplots(figsize=(10, 7))

    for datum, label in zip(enc_1_biases_metric_epoch.T, enc_labels[:-1]):
        ax.scatter(epochs, datum, alpha=.75, label=label)

    for datum, label in zip(dec_1_biases_metric_epoch.T, dec_labels[:-1]):
        ax.scatter(epochs, datum, alpha=.75, label=label)

    ax.set_xlabel("Inter-Epoch $i$", size=20)
    ax.set_ylabel("$||W_{i + 1} - W_i||_2/||W_i||_2$", size=20)

    ax.tick_params(axis='both', length=10, labelsize=15)

    ax.legend(loc='best', fontsize=10, ncol=4)
    ax.grid(True)
    ax.set_title(f"{current_model.capitalize()}-single; {epochs[-1]+1} Epochs; Biases; {run_time}.".replace("_", " "))

    fig.savefig(f"{weight_comp_path}\\euclidean_biases_plot.jpg", dpi=400)

    ax.set_yscale('log')
    ax.set_ylabel("$\\log_{10 }||W_{i + 1} - W_i||_2/||W_i||_2$", size=20)
    fig.savefig(f"{weight_comp_path}\\log_plots\\euclidean_biases_plot_log.jpg", dpi=400)

    plt.close(fig)


print(f"That took: {dt.datetime.now() - beginning_time}.")