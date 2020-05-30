import numpy as np
import os
from keras.preprocessing import sequence


def __normalize_mfcc(file_name, feats_mean, feats_std, eps=1e-14):
    feature = np.load(file_name)['x']

    return (feature - feats_mean) / (feats_std + eps)


def generate_mfcc_features(pd,mfcc_path,feat_norm_file):
                          
    # prepare mfcc features
    feats_std = np.load(feat_norm_file)['std']
    feats_mean = np.load(feat_norm_file)['mean']

    mfccs = []
    for index, name in enumerate (pd):
        #print ("preprocessing: ", index, " : ", name)
        mfcc_file_name = mfcc_path+name+"_mfcc_no_cvmn.npz"
        if not os.path.isfile(mfcc_file_name):
            print ("ERROR: Missin MFCC ", mfcc_file_name)
            continue
        mfccs.append(__normalize_mfcc(mfcc_file_name, feats_mean, feats_std))

    max_length = min(2000, max([len(f) for f in mfccs]))

    max_length = 2000 # hyper parameter to tune later
    return sequence.pad_sequences(mfccs, maxlen=max_length, dtype='float32')

