import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t

def linear_model(x,a,b):
    return a*x+b

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

        ax.plot(x, y, '-', lw=2.5, color=color, label=label)
        #ax.fill_between(x, y+d_mean, y-d_mean, alpha=.20, fc=color)
        #ax.fill_between(x, y_ru, y_rl, alpha=.20, fc=color)#, label=r"95% RV interval")
        return opt_params
    except:
        print(f"Fit failure for {model_name}, {column_name}, {lift_dim}, {label}.")
        ax.plot([0], [0], '-', lw=2.5, color=color, label=label+" FF")
        return None

examples_folder = r"C:\Users\josep\Documents\School Work\Research\Curtis\Machine_Learning\my_work\DLDMD-newest\examples"
models = os.listdir(examples_folder)
model_folders = [os.path.join(examples_folder, model, "trained_models") for model in models]
figures_and_axes = None

transient_percent = .2

for model_folder, model_name in zip(model_folders, models):

    # [os.remove(os.path.join(examples_folder, model_name, file)) for file in os.listdir(os.path.join(examples_folder, model_name)) if file.endswith('.png')]

    individual_runs = os.listdir(model_folder)[:8]
    individual_runs_folder = [os.path.join(model_folder, individual_run) for individual_run in individual_runs]
    color_map = plt.cm.get_cmap('tab20b', len(individual_runs))
    colors = color_map([qq for qq in range(len(individual_runs))])

    lat_dims = []

    columns_dict = None

    for color, dim_run, dim_run_folder in zip(colors, individual_runs, individual_runs_folder):

        data_file = os.path.join(dim_run_folder, "divergence_results.csv")
        # loss_file = os.path.join(dim_run_folder, "losses_by_epoch.csv")
        
        if not os.path.exists(data_file):
            continue
        data_frame = pd.read_csv(data_file, index_col=0)
        if data_frame.shape[0] < 250:
            continue

        fold_components = dim_run.split('_')
        lat_dim = fold_components[-2]
        run_time = fold_components[-1]

        if os.path.exists(os.path.join(dim_run_folder, "loss_plot.png")):
            os.remove(os.path.join(dim_run_folder, "loss_plot.png"))

            # loss_frame = pd.read_csv(loss_file, index_col=0)

            # epoch = loss_frame.index.values
            # losses = loss_frame[loss_frame.columns[0]]

            # figure, axes = plt.subplots(figsize=(10,7))

            # axes.plot(epoch, losses, '-k')

            # figure.tight_layout()
            # figure.savefig(os.path.join(dim_run_folder, "loss_plot.png"))
            # plt.close(figure)


        lat_dims.append(lat_dim)

        inter_epoch = np.array(data_frame.index)
        cut_off = int(inter_epoch.size*transient_percent)
        inter_epoch = inter_epoch[cut_off:]

        if figures_and_axes is None:
            figures_and_axes = [(_, *plt.subplots(figsize=(10,7))) for _ in data_frame.columns]

        if columns_dict is None:
            columns_dict = {column_name: [] for column_name in data_frame.columns}
        
        for (column_name, fig, ax) in figures_and_axes:
            datum = np.log10(data_frame[column_name].values)[cut_off:]

            params = local_fit(linear_model, inter_epoch, datum, ax, color, model_name, 
                               column_name, lat_dim, f"Lift Dim: {lat_dim}")
            columns_dict[column_name].append(params[0])

    lat_dims = np.array(lat_dims)

    slope_frame = pd.DataFrame(columns_dict, index=lat_dims)
    slope_frame.to_csv(os.path.join(examples_folder, model_name, "linear_fit_slopes.csv"))
    
    slope_data = slope_frame.values
    average_slope_by_dim = slope_data.mean(axis=1)
    variance_by_dim = np.sqrt(slope_data.var(axis=1))

    fig, ax = plt.subplots(figsize=(10,7))

    ax.grid(True, which='both')
    ax.bar(lat_dims, average_slope_by_dim, bottom=0, color='tab:red', yerr=variance_by_dim, error_kw={'capthick': 50})
    # ax.bar(lat_dims, average_slope_by_dim+variance_by_dim, bottom=average_slope_by_dim-variance_by_dim, color='tab:blue', alpha=.6)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
    ax.set_xlabel("Lifting Dimension", size=15)
    ax.set_ylabel("Average slope of linear fit of $\\log_{10}$ Divergences", size=15)
    ax.tick_params(axis='both', length=9, labelsize=12.5)        
    ax.set_title(f"Model: {model_name}", size=20)

    fig.tight_layout()
    fig.savefig(os.path.join(examples_folder, model_name, f"slope_linear_fit.png"))

    plt.close(fig)


    columns_dict = None

    for column_name in slope_frame.columns:
        datum = slope_frame[column_name]

        fig, ax = plt.subplots(figsize=(10,7))
        ax.grid(True, which='both')
        ax.bar(lat_dims, datum, bottom=0)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
        ax.set_xlabel("Lifting Dimension", size=15)
        ax.set_ylabel("Slope of linear fit of $\\log_{10}$ Divergences", size=15)
        ax.tick_params(axis='both', length=9, labelsize=12.5)        
        ax.set_title(f"Model: {model_name}, Layer: {column_name}", size=20)

        fig.tight_layout()
        fig.savefig(os.path.join(examples_folder, model_name, f"slope_linear_fit_{column_name}.png"))

        plt.close(fig)


    lat_dims = np.array([int(qq) for qq in lat_dims])
    norm = plt.Normalize(vmin=lat_dims[0]-.5, vmax=lat_dims[-1]+.5)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map)

    for (column_name, fig, ax) in figures_and_axes:
        ax.grid(True, which='both')
        ax.set_xlabel("Inter-epoch", size=15)
        ax.set_ylabel("$\log_{10}$ Divergences", size=15)
        ax.tick_params(axis='both', length=9, labelsize=12.5)
        # ax.legend(loc='best', fontsize=10, ncol=5)
        ax.set_title(f"Model: {model_name}, Layer: {column_name}", size=20)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        c_bar = fig.colorbar(sm, cax=cax, ticks=lat_dims)
        c_bar.ax.get_yaxis().labelpad = 15
        c_bar.ax.set_ylabel("Lifting Dimension", rotation=270, size=15)
        #c_bar.set_ticks(lat_dims)
        #c_bar.set_ticklabels(lat_dims)

        fig.tight_layout()
        fig.savefig(os.path.join(examples_folder, model_name, f"div_plot_linear_{column_name}.png"))

        plt.close(fig)


    figures_and_axes = None