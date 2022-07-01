import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.neighbors import KernelDensity
from scipy import integrate

def kde_from_data(data_set, kde_kernel='exponential', 
                  kde_bandwidth=.4, discrete_points=1001):
    
    set_max = data_set.max()
    set_min = data_set.min()

    new_x = np.linspace(set_min, set_max, discrete_points)[:, np.newaxis]

    kde = KernelDensity(kernel=kde_kernel, bandwidth=kde_bandwidth).fit(data_set.reshape((data_set.size, 1)))

    pdf = np.exp(kde.score_samples(new_x))

    simpson_area = integrate.simpson(pdf, new_x[:,0])
    
    
    pdf /= simpson_area

    return new_x, pdf


current_model = 'duffing'
first_cutoff = len(current_model) + 1

model_folder = "C:\\Users\\josep\\Documents\\School Work\\Research\\Curtis\\Machine_Learning" + \
                    f"\\my_work\\DLDMD-newest\\weight_comp\\{current_model}-single"

                    
file_list = np.array([file for file in os.listdir(model_folder) if file.startswith(current_model)])

lifting_dims = np.array([int(file[first_cutoff:-16]) for file in file_list])

positions = np.argsort(lifting_dims)

file_list = file_list[positions]
lifting_dims = lifting_dims[positions]