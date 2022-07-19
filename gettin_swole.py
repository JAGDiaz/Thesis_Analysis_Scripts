import os
import sys
import shutil

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

model_folders = [os.path.join(examples_folder, model) for model in [file for file in os.listdir(examples_folder) if os.path.splitext(file)[-1] == '']]

for model_folder in model_folders:
    print(model_folder.split('\\')[-1])

    trained_folder = os.path.join(model_folder, "trained_models")

    for dim_run in os.listdir(trained_folder):

        model_weight_folder = os.path.join(trained_folder, dim_run)
        weight_files = [os.path.join(model_weight_folder, file) for file in os.listdir(model_weight_folder) if file.endswith('.h5')]

        if len(weight_files) == 0:
            continue

        new_home = os.path.join(model_weight_folder, "weights_by_epoch")

        if not os.path.exists(new_home):
            os.mkdir(new_home)
        
        new_file_locations = [os.path.join(new_home, file) for file in os.listdir(model_weight_folder) if file.endswith('.h5')]

        for ii, (weight_file, new_weight_file) in enumerate(zip(weight_files, new_file_locations)):
            shutil.move(weight_file, new_weight_file)
            printProgressBar(ii, dim_run, len(weight_files))
