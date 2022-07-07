import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t

def nonlinear_model(x,a,b,c):
    return -a*(x*x)/(x*x + b) + c

def og_model(j, c, p, d):
    return c/(j**p) + d

def linear_model(x,a,b):
    return a*x+b

def nonlinear_2(x,a,b):
    return a*x/np.sqrt(1+x**2) + b

def exponential(x,a,b,c):
    return a*np.exp(-b*x*x) + c

def exp_2(x,a,b,c):
    return a*x**b + c

def regression_CIs(xd, yd, model, conf=0.95):#, num_points=1_000):
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

def local_fit(model, epics, data, ax, color, model_name, column_name, lift_dim, label):
    try:
        y, y_l, y_u, x, y_rl, y_ru, opt_params = \
            regression_CIs(epics, data, model)
        d_mean = np.mean(np.abs(y - data))

        ax.plot(x, y, '--', lw=2.5, color=color, label=label)
        ax.fill_between(x, y+d_mean, y-d_mean, alpha=.20, fc=color)
        ax.fill_between(x, y_ru, y_rl, alpha=.20, fc=color)#, label=r"95% RV interval")
        return opt_params
    except:
        print(f"Fit failure for {model_name}, {column_name}, {lift_dim}, {label}.")
        ax.plot([0], [0], '--', lw=2.5, color=color, label=label+" FF")
        return None

examples_folder = os.path.join(os.getcwd(), "examples")

models = os.listdir(examples_folder)

model_folders = [os.path.join(examples_folder, model) for model in models]

model_dict = {"nonlinear": nonlinear_model, "og_model": nonlinear_model, "linear": linear_model,
              "nonlinear2": nonlinear_2, "exponential": exponential, "exp_2": exp_2}
eqn_dict = {"nonlinear": "$f(x)=-\\frac{a x^2}{x^2 + b} + c$", 
            "og_model": "$f(x)=\\frac{a}{x^b}+c$", 
            "linear": "$f(x)=ax+b$", 
            "nonlinear2":"$f(x) = \\frac{ax}{\\sqrt{1+x^2}} + b$",
            "exponential": "$f(x) = a\\exp(-bx^2)+c$",
            "exp_2": "$f(x) = ax^b + c$"}
keys = list(model_dict.keys())
num_rows = len(keys)//2 if (not len(keys) % 2) else (len(keys)//2 + 1)
# colors = ['cyan', 'purple', 'red', 'brown', 'hotpink']

transient_percent = .2

#fitting_func = "linear"
#model_to_fit = model_dict[fitting_func]

for model, model_name in zip(model_folders, models):
    print(f"Generating plots for {model_name}.")
    trained_folder = os.path.join(model, "trained_models")

    for dim_run in os.listdir(trained_folder):
        
        fold_components = dim_run.split('_')
        lat_dim = fold_components[-2]
        run_time = fold_components[-1]

        divs_folder = os.path.join(trained_folder, dim_run)
        
        [os.remove(os.path.join(divs_folder,file)) for file in os.listdir(divs_folder) if file.endswith('.png')]
        divs_file = os.path.join(divs_folder, "divergence_results.csv")

        if not os.path.exists(divs_file):
            continue

        data_frame = pd.read_csv(divs_file, index_col=0)

        inter_epoch = np.array(data_frame.index)
        cut_off = int(inter_epoch.size*transient_percent)
        inter_epoch = inter_epoch[cut_off:]

        for col in data_frame.columns:

            datum = np.log10(data_frame[col].values)[cut_off:]

            # fig, axes = plt.subplots(num_rows, 2, figsize=(9*2, 7*num_rows), sharex='col')
            # axes = axes.flatten()

            # for ax, key in zip(axes, keys):
                # ax.plot(inter_epoch, datum, '.k')
                # # for jj in range(1, 5):
                    # # try:
                        # # coeffs = np.polyfit(inter_epoch, datum, jj)
                        # # fit_func = lambda ex: sum([coef*ex**ii for ii, coef in enumerate(coeffs[::-1])])
                        # # ax.plot(inter_epoch, fit_func(inter_epoch), '-', label=f"{jj} degree fit")
                    # # except:
                        # # print(f"Fit failure for {jj} degree: {model_name}, {lat_dim}, {col}")
                # local_fit(model_dict[key], inter_epoch, datum, ax, 'red', model_name, 
                        # col, lat_dim, eqn_dict[key])
                # # ax.set_yscale('log')

                # ax.grid(True, which='both')
                # ax.set_ylabel("$\log_{10}$ Divergences", size=15)
                # ax.tick_params(axis='y', length=9, labelsize=12.5)
                # ax.tick_params(axis='x', length=0)
                # ax.legend(loc='best', fontsize=20)

            # fig.suptitle(f"Model: {model_name}, Lat_dim: {lat_dim}, Layer: {col}", size=25, y=.99)
            # axes[-1].set_xlabel("Inter-epoch", size=15)
            # fig.tight_layout()
            # fig.savefig(f"{divs_folder}\\div_plot_{col}.png")
            # plt.close(fig)
            # del fig, axes, datum

            fig, ax = plt.subplots(figsize=(10, 7))
            
            ax.plot(inter_epoch, datum, '.k')
            params = local_fit(model_dict['linear'], inter_epoch, datum, ax, 'red', model_name, 
                               col, lat_dim, eqn_dict['linear'])

            param_string = ", ".join([f"{chr(jj)} = {params[kk]:.2e}" for kk, jj in enumerate(range(97, 97+len(params)))]) \
                if params is not None else ""

            ax.grid(True, which='both')
            ax.set_xlabel("Inter-epoch", size=15)
            ax.set_ylabel("$\log_{10}$ Divergences", size=15)
            ax.tick_params(axis='both', length=9, labelsize=12.5)
            ax.legend(loc='best', fontsize=17.5)
            ax.set_title(f"Model: {model_name}, Lat_dim: {lat_dim}, Layer: {col},\n {param_string}", size=20)

            fig.tight_layout()
            fig.savefig(f"{divs_folder}\\div_plot_linear_{col}.png")
            plt.close(fig)
            del fig, ax, datum