import sys
import os
import numpy as np
import glob

np.random.seed(1337)  # for reproducibility


def compute_mean_var_large_scale_maximum_likelihood(folder_path):

    N_MFCC_COEFFS = 13
    N = 0
    mean_acc = np.zeros((1, N_MFCC_COEFFS), dtype='float32')
    cov_acc = np.zeros((N_MFCC_COEFFS, N_MFCC_COEFFS), dtype='float32')
    for filepath in sorted(glob.glob(folder_path)):
        print("processing file: ", filepath)
        data_file = (np.load(filepath)['x'])
        mean_acc = mean_acc+np.sum(data_file, axis=0)
        cov_acc = cov_acc + (data_file.T).dot(data_file)
        N = N+data_file.shape[0]

    mean = mean_acc/float(N)
    cov = cov_acc/float(N) - (mean.T).dot(mean)
    return mean.flatten(),  np.sqrt(cov.diagonal())


_out_folder = sys.argv[1]
train_folder_path = _out_folder + '/*mfcc*npz'


# mean,std=compute_mean_var_small_scale(train_folder_path)
mean, std = compute_mean_var_large_scale_maximum_likelihood(train_folder_path)
np.savez(_out_folder+"/mean_std.npz", mean=mean, std=std)