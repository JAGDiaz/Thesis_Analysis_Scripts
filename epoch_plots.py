import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit, least_squares
from scipy.stats import t, linregress
#plt.rcParams['text.usetex'] = True


def regression_CIs(xd, yd, model, conf=0.95, num_points=1_000):
    alpha=1.-conf   # significance
    n = xd.size   # data sample size
        
    # Predicted values from fitted model:
    opt_params, opt_pcov = curve_fit(model, xd, yd)
    y = model(xd, *opt_params)
    
    var = 1./(n-2.)*np.sum((yd-model(xd, *opt_params))**2) # Variance
    sd = np.sqrt(var) # Standard Deviation
    sxd = np.sum((xd-xd.mean())**2)
    sx  = (xd-xd.mean())**2
    
    # quantile of student's t distribution for p=1-alpha/2
    q = t.ppf(1.-alpha/2, n-2)
    
    # get the upper and lower CI:
    dy = q*sd*np.sqrt(1./n + sx/sxd)
    dr = q*sd*np.sqrt(1 + 1./n + sx/sxd)
    rl = y - dr
    ru = y + dr
    yl = y-dy
    yu = y+dy
    
    return y, yl, yu, xd, rl, ru, opt_params

def model_to_fit(j, c, p, d):
    return c/(j**p) + d

def other_func(params, x, y):
    return np.log(y - params[0]) - np.log(params[1]) + params[2]*np.log(x)

def total_variation(f_vals):
    return np.cumsum(np.abs(np.diff(f_vals)))

def one_step_derivative(x_vals, f_vals):
    return np.diff(f_vals)/np.diff(x_vals)

def fit_data(pd_dataframe, model, file_label, y_label, folder_loc, the_model, run_time, fit=True):

    epics = np.arange(pd_dataframe.shape[0]) + 1

    for column_name in pd_dataframe.columns:

        data = pd_dataframe[column_name]

        fig = plt.figure(figsize=(30,.5625*30))
        fig.suptitle(f"{column_name} ".replace("_", " ").capitalize() + f"{file_label} {the_model}-{run_time}".capitalize(), size=30)

        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 2, 3)
        ax3 = fig.add_subplot(2, 2, 4)

        ax1.plot(epics, data, 'k.', ms=10, label="Raw Data")

        if fit:
            try:
                y, y_l, y_u, x, y_rl, y_ru, opt_params = \
                    regression_CIs(epics, data, model)
                ax1.plot(x, y, '-', lw=5, color='purple')
                ax1.fill_between(x, y_u, y_l, alpha=.25, fc='red', label=r"95% confidence interval")
                ax1.fill_between(x, y_ru, y_rl, alpha=.25, fc='blue', label=r"95% RV interval")
                ax1.set_title("$M_j = \\frac{c }{j^p} + d$ for $c = %6.5f,\\ p = %6.5f,\\ d = %6.5f$" % tuple(opt_params), size=25)
            except:
                print(f"Fit failure for {file_label}, {column_name}.")

        ax1.set_xlabel("Inter-epoch $i$", size=25)
        ax1.set_ylabel(y_label, size=25)
        ax1.set_yscale("log")
        ax1.tick_params(axis='both', which='both', labelsize=20)
        ax1.grid(True, which='both')
        ax1.legend(loc='best', fontsize=15)

        ax2.plot(epics[:-1], total_variation(data), 'k-', lw=5)
        ax2.set_xlabel("Inter-epoch $i$", size=25)
        ax2.set_ylabel("$T$", size=25)
        ax2.set_title("Running total variation by Inter-Epoch.", size=25)
        ax2.tick_params(axis='both', which='both', labelsize=20)
        ax2.grid(True, which='both')

        ax3.plot(epics, np.gradient(data, epics), 'k.', lw=5)
        ax3.set_xlabel("Inter-epoch $i$", size=25)
        ax3.set_ylabel("$D$", size=25)
        ax3.set_title("Numerical Derivative", size=25)
        ax3.tick_params(axis='both', which='both', labelsize=20)
        ax3.grid(True, which='both')

        fig.tight_layout(pad=2)
        fig.savefig(folder_loc+f"\\{file_label}_{column_name}_comp.jpg", dpi=400)
        plt.close(fig)



if __name__ == '__main__':    

    current_model = 'van_der_pol'
    running_time = '2021-11-16-0807_8'

    data_folder = "C:\\Users\\josep\\Documents\\School Work\\Research\\Curtis\\Machine_Learning" + \
                    f"\\my_work\\DLDMD-newest\\weight_comp\\{current_model}-single\\{running_time}"



    # Euclid weights

    enc_diff_weight_euclids = pd.read_csv(data_folder+"\\encoder_diff_weights_euclids.csv", index_col=0) 
    fit_data(enc_diff_weight_euclids, model_to_fit, "euclid_diff_weights", "$\\log_{10 }||W_{i + 1} - W_i||_2$", data_folder,
        current_model, running_time)
        
    enc_1_weight_euclids = pd.read_csv(data_folder+"\\encoder_1_weights_euclids.csv", index_col=0) 
    fit_data(enc_diff_weight_euclids/enc_1_weight_euclids, model_to_fit, "euclid_1_weights", "$\\log_{10 }||W_{i + 1} - W_i||_2/||W_i||_2$", data_folder,
        current_model, running_time)
    fit_data(enc_1_weight_euclids, model_to_fit, "euclid_size_weights", "$\\log_{10 }||W_i||_2$", data_folder,
        current_model, running_time, fit=False)

    enc_2_weight_euclids = pd.read_csv(data_folder+"\\encoder_2_weights_euclids.csv", index_col=0) 
    fit_data(enc_diff_weight_euclids/enc_2_weight_euclids, model_to_fit, "euclid_2_weights", "$\\log_{10 }||W_{i + 1} - W_i||_2/||W_{i + 1}||_2$", data_folder,
        current_model, running_time)

    dec_diff_weight_euclids = pd.read_csv(data_folder+"\\decoder_diff_weights_euclids.csv", index_col=0)
    fit_data(dec_diff_weight_euclids, model_to_fit, "euclid_diff_weights", "$\\log_{10 }||W_{i + 1} - W_i||_2$", data_folder,
        current_model, running_time)

    dec_1_weight_euclids = pd.read_csv(data_folder+"\\decoder_1_weights_euclids.csv", index_col=0)
    fit_data(dec_diff_weight_euclids/dec_1_weight_euclids, model_to_fit, "euclid_1_weights", "$\\log_{10 }||W_{i + 1} - W_i||_2/||W_i||_2$", data_folder,
        current_model, running_time)
    fit_data(dec_1_weight_euclids, model_to_fit, "euclid_size_weights", "$\\log_{10 }||W_i||_2$", data_folder,
        current_model, running_time, fit=False)

    dec_2_weight_euclids = pd.read_csv(data_folder+"\\decoder_2_weights_euclids.csv", index_col=0)
    fit_data(dec_diff_weight_euclids/dec_2_weight_euclids, model_to_fit, "euclid_2_weights", "$\\log_{10 }||W_{i + 1} - W_i||_2/||W_{i + 1}||_2$", data_folder,
        current_model, running_time)



    # Euclid Biases
    enc_diff_biases_euclids = pd.read_csv(data_folder+"\\encoder_diff_biases_euclids.csv", index_col=0) 
    fit_data(enc_diff_biases_euclids, model_to_fit, "euclid_diff_biases", "$\\log_{10 }||B_{i + 1} - B_i||_2$", data_folder,
        current_model, running_time)

    enc_1_biases_euclids = pd.read_csv(data_folder+"\\encoder_1_biases_euclids.csv", index_col=0) 
    fit_data(enc_diff_biases_euclids/enc_1_biases_euclids, model_to_fit, "euclid_1_biases", "$\\log_{10 }||B_{i + 1} - B_i||_2/||B_i||_2$", data_folder,
        current_model, running_time)
    fit_data(enc_1_biases_euclids, model_to_fit, "euclid_size_biases", "$\\log_{10 }||B_i||_2$", data_folder,
        current_model, running_time, fit=False)

    enc_2_biases_euclids = pd.read_csv(data_folder+"\\encoder_2_biases_euclids.csv", index_col=0) 
    fit_data(enc_diff_biases_euclids/enc_2_biases_euclids, model_to_fit, "euclid_2_biases", "$\\log_{10 }||B_{i + 1} - B_i||_2/||B_{i + 1}||_2$", data_folder,
        current_model, running_time)

    dec_diff_biases_euclids = pd.read_csv(data_folder+"\\decoder_diff_biases_euclids.csv", index_col=0)
    fit_data(dec_diff_biases_euclids, model_to_fit, "euclid_diff_biases", "$\\log_{10 }||B_{i + 1} - B_i||_2$", data_folder,
        current_model, running_time)

    dec_1_biases_euclids = pd.read_csv(data_folder+"\\decoder_1_biases_euclids.csv", index_col=0)
    fit_data(dec_diff_biases_euclids/dec_1_biases_euclids, model_to_fit, "euclid_1_biases", "$\\log_{10 }||B_{i + 1} - B_i||_2/||B_i||_2$", data_folder,
        current_model, running_time)
    fit_data(dec_1_biases_euclids, model_to_fit, "euclid_size_biases", "$\\log_{10 }||B_i||_2$", data_folder,
        current_model, running_time, fit=False)

    dec_2_biases_euclids = pd.read_csv(data_folder+"\\decoder_2_biases_euclids.csv", index_col=0)
    fit_data(dec_diff_biases_euclids/dec_2_biases_euclids, model_to_fit, "euclid_2_biases", "$\\log_{10 }||B_{i + 1} - B_i||_2/||B_{i + 1}||_2$", data_folder,
        current_model, running_time)




    # Divergence weights
    enc_weight_divs = pd.read_csv(data_folder+"\\encoder_weights_divergences.csv", index_col=0)
    fit_data(enc_weight_divs, model_to_fit, "div_weights", "$\\log_{10 }KL(p_{i+1} || p_i)$", data_folder,
        current_model, running_time, fit=False)

    dec_weight_divs = pd.read_csv(data_folder+"\\decoder_weights_divergences.csv", index_col=0)
    fit_data(dec_weight_divs, model_to_fit, "div_weights", "$\\log_{10 }KL(p_{i+1} || p_i)$", data_folder,
        current_model, running_time, fit=False)


    # Divergence biases
    enc_weight_divs = pd.read_csv(data_folder+"\\encoder_biases_divergences.csv", index_col=0)
    fit_data(enc_weight_divs, model_to_fit, "div_biases", "$\\log_{10 }KL(p_{i+1} || p_i)$", data_folder,
        current_model, running_time, fit=False)

    dec_weight_divs = pd.read_csv(data_folder+"\\decoder_biases_divergences.csv", index_col=0)
    fit_data(dec_weight_divs, model_to_fit, "div_biases", "$\\log_{10 }KL(p_{i+1} || p_i)$", data_folder,
        current_model, running_time, fit=False)



