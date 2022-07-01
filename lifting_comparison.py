import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from epoch_plots import fit_data, model_to_fit, regression_CIs

def local_fit(model, epics, data, axes, color, file_label, column_name, lift_dim, label):
    try:
        y, y_l, y_u, x, y_rl, y_ru, opt_params = \
            regression_CIs(epics, data, model)

        d_mean = np.mean(np.abs(y - data))

        for ax, col in zip(axes, [color, 'blue']):
            ax.plot(x, y, '--', lw=2.5, color=col,label=label)
            ax.fill_between(x, y+d_mean, y-d_mean, alpha=.20, fc=col)
        #axes.fill_between(x, y_ru, y_rl, alpha=.20, fc=color)#, label=r"95% RV interval")
    except:
        print(f"Fit failure for {file_label}, {column_name}, {lift_dim}.")
        for ax, col in zip(axes, [color, 'blue']):
            ax.plot([0], [0], '--', lw=2.5, color=col,label=label+" FF")


current_model = 'van_der_pol'
first_cutoff = len(current_model) + 1

model_folder = "C:\\Users\\josep\\Documents\\School Work\\Research\\Curtis\\Machine_Learning" + \
                    f"\\my_work\\DLDMD-newest\\weight_comp\\{current_model}-single"

file_list = np.array([file for file in os.listdir(model_folder) if file.startswith(current_model)])

lifting_dims = np.array([int(file[first_cutoff:-16]) for file in file_list])

positions = np.argsort(lifting_dims)

num_curves = 4
file_list = file_list[positions][:num_curves]
lifting_dims = lifting_dims[positions][:num_curves]

if num_curves < 5:
    colors = ['black', 'red', 'green', 'blue'][:num_curves]
else:
    color_map = plt.get_cmap('jet')

    colors = [color_map(ii/num_curves) for ii in range(num_curves)]


layer_list = ['enc_in', 'enc_0', 'enc_1', 'enc_2', 'enc_out', 'dec_in', 'dec_0', 'dec_1', 'dec_2', 'dec_out']

metric = ['weights_divergences', 'diff_weights_euclids']

layer_role = ['encoder', 'decoder']

for met in metric:

    for layer_r in layer_role:

        for layer in layer_list:

            fig1, ax1 = plt.subplots(figsize=(16,9))

            for lift, file_dir, color in zip(lifting_dims, file_list, colors):
                data_folder = "C:\\Users\\josep\\Documents\\School Work\\Research\\Curtis\\Machine_Learning" + \
                            f"\\my_work\\DLDMD-newest\\weight_comp\\{current_model}-single\\{file_dir}"

                diff_weight_euclids = pd.read_csv(data_folder+f"\\{layer_r}_{met}.csv", index_col=0) 

                try:
                    data = diff_weight_euclids[layer]
                except:
                    continue

                epochs = np.arange(diff_weight_euclids.shape[0]) + 1
                data = np.log10(data)

                fig2, ax2 = plt.subplots(figsize=(16,9))

                # Euclid weights

                ax2.plot(epochs, data, '.', color='k')
                local_fit(model_to_fit, epochs, data, [ax1, ax2], color, file_dir, layer, lift, 
                        f"Lifting dimension: {lift}")

                ax2.grid(True, which='both')
                ax2.set_title(f"{current_model}, {met}, Layer: {layer}", size=20)

                ax2.set_ylabel("$\\log_{10 }||W_{i + 1} - W_i||_2$", size=15)
                ax2.set_xlabel("Inter Epoch $i$", size=15)

                ax2.legend(loc='best', ncol=3, fontsize=10, framealpha=1)

                fig2.tight_layout()
                fig2.savefig(f".\\examples\\{current_model}\\lifting_{layer_r}_{lift}_{met}_{layer}.jpg")
                plt.close(fig2)
                
            
            
            #ax.set_yscale('log')
            ax1.grid(True, which='both')
            ax1.set_title(f"{current_model}, {met}, Layer: {layer}", size=20)

            ax1.set_ylabel("$\\log_{10 }||W_{i + 1} - W_i||_2$", size=15)
            ax1.set_xlabel("Inter Epoch $i$", size=15)

            ax1.legend(loc='best', ncol=3, fontsize=10, framealpha=1)

            fig1.tight_layout()
            fig1.savefig(f".\\examples\\{current_model}\\lifting_comparison_{layer_r}_{met}_{layer}.jpg")
            plt.close(fig1)














    # enc_1_weight_euclids = pd.read_csv(data_folder+"\\encoder_1_weights_euclids.csv", index_col=0) 

    # enc_2_weight_euclids = pd.read_csv(data_folder+"\\encoder_2_weights_euclids.csv", index_col=0) 

    # dec_diff_weight_euclids = pd.read_csv(data_folder+"\\decoder_diff_weights_euclids.csv", index_col=0)

    # dec_1_weight_euclids = pd.read_csv(data_folder+"\\decoder_1_weights_euclids.csv", index_col=0)

    # dec_2_weight_euclids = pd.read_csv(data_folder+"\\decoder_2_weights_euclids.csv", index_col=0)


    # # Euclid biases
    # enc_diff_biases_euclids = pd.read_csv(data_folder+"\\encoder_diff_biases_euclids.csv", index_col=0) 

    # enc_1_biases_euclids = pd.read_csv(data_folder+"\\encoder_1_biases_euclids.csv", index_col=0) 

    # enc_2_biases_euclids = pd.read_csv(data_folder+"\\encoder_2_biases_euclids.csv", index_col=0) 

    # dec_diff_biases_euclids = pd.read_csv(data_folder+"\\decoder_diff_biases_euclids.csv", index_col=0)

    # dec_1_biases_euclids = pd.read_csv(data_folder+"\\decoder_1_biases_euclids.csv", index_col=0)

    # dec_2_biases_euclids = pd.read_csv(data_folder+"\\decoder_2_biases_euclids.csv", index_col=0)


    # # Divergence weights
    # enc_weight_divs = pd.read_csv(data_folder+"\\encoder_weights_divergences.csv", index_col=0)

    # dec_weight_divs = pd.read_csv(data_folder+"\\decoder_weights_divergences.csv", index_col=0)


    # # Divergence biases
    # enc_weight_divs = pd.read_csv(data_folder+"\\encoder_biases_divergences.csv", index_col=0)

    # dec_weight_divs = pd.read_csv(data_folder+"\\decoder_biases_divergences.csv", index_col=0)
