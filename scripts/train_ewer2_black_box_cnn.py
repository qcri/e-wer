import data
import models
import mfcc 
import numpy as np
from keras.layers import Input
from keras.layers import Dense
from keras.layers import concatenate
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import sys

# Copyright 2020  QCRI, HBKU (Author: Ahmed Ali)
# Apache 2.0.


train="data/dev_mgb2.feats"
dev="data/eval_mgb2.feats"
test="data/summa_ar.feats"
_EPOCHS=50


train_data = data.load_features (train,data.cols)
dev_data = data.load_features (dev,data.cols)
test_data = data.load_features (test,data.cols)


####
# Load the acoustics data
####

#Load train data
train_acoustics_path="data/wav_splits/segments_dev_mgb2/"
train_mfcc = mfcc.generate_mfcc_features (train_data["id"],train_acoustics_path,train_acoustics_path+"mean_std.npz")
print (train_mfcc.shape)

#Load dev data 
dev_acoustics_path="data/wav_splits/segments_eval_mgb2/"
dev_mfcc = mfcc.generate_mfcc_features (dev_data["id"],dev_acoustics_path,dev_acoustics_path+"mean_std.npz")
print (dev_mfcc.shape)

#load test data
test_acoustics_path="data/wav_splits/segments_eval_summa/"
test_mfcc = mfcc.generate_mfcc_features (test_data["id"],test_acoustics_path,test_acoustics_path+"mean_std.npz")
print (test_mfcc.shape)



_feature="phoneme"
_ngram=3
word_list_p = data.return_word_list (train_data[_feature].values.tolist(),_ngram)
train_feats_p = data.make_features (train_data[_feature].values.tolist(), word_list_p, _ngram, True)
dev_feats_p = data.make_features (dev_data[_feature].values.tolist(), word_list_p, _ngram, True)
test_feats_p = data.make_features (test_data[_feature].values.tolist(), word_list_p, _ngram,  True)


_feature="words"
_ngram=2
word_list_w = data.return_word_list (train_data[_feature].values.tolist(), _ngram)
train_feats_w = data.make_features (train_data[_feature].values.tolist(), word_list_w, _ngram, True)
dev_feats_w = data.make_features (dev_data[_feature].values.tolist(), word_list_w, _ngram, True)
test_feats_w = data.make_features (test_data[_feature].values.tolist(), word_list_w, _ngram, True)

print (len(word_list_w), len(word_list_p),train_feats_p.shape[1],train_feats_w.shape[1])
max_features_p = len(word_list_p)
max_features_w = len(word_list_w)


## call back 
mlp = models.create_mlp(train_data[data.continuous_black].shape[1])
cnn_acoustics = models.create_acoustics_cnn ()
cnn_phoneme = models.create_1d_txt_cnn(sequence_length=train_feats_p.shape[1],vocabulary_size=max_features_p,embedding_dim=256)
cnn_words = models.create_1d_txt_cnn(sequence_length=train_feats_w.shape[1],vocabulary_size=max_features_w,embedding_dim=256)

#combinedInput = mlp.output
combinedInput = concatenate([mlp.output, cnn_acoustics.output, cnn_phoneme.output, cnn_words.output])
wer = Dense(32, activation="relu")(combinedInput)
wer = Dropout(0.2) (wer)

wer = Dense(1, kernel_initializer='normal')(wer)

final_model='eWER2_black_box_cnn'

model = Model(inputs=[mlp.input, cnn_acoustics.input, cnn_phoneme.input, cnn_words.input], outputs= wer)
plot_model(model, to_file='results/'+final_model+'_plots.pdf', show_shapes=True, show_layer_names=True)
model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])


# TRAIN E-WER
'''
#earlystopper = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=40, patience=1, verbose=1, mode='auto')
earlystopper = EarlyStopping(monitor='val_loss', min_delta=0.0001,
                             patience=1, verbose=1, mode='auto')

checkpoint = ModelCheckpoint(filepath=final_model+'.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', 
                             verbose=1, save_best_only=True, mode='auto')

model.fit([train_data[data.continuous_glass], train_mfcc, train_feats_p, train_feats_w], train_data["wer"].to_numpy(),
           validation_data=([dev_data[data.continuous_glass], dev_mfcc, dev_feats_p, dev_feats_w], dev_data["wer"].to_numpy()),
           batch_size=32,  epochs=_EPOCHS, verbose=1, 
           callbacks=[checkpoint,earlystopper])  
'''           
model.summary()

              
model.fit([train_data[data.continuous_black], train_mfcc, train_feats_p, train_feats_w], train_data["wer"].to_numpy(),
           batch_size=32,  epochs=_EPOCHS, verbose=1)
model.save('results/'+final_model+'.h5', overwrite=True)


model = load_model('results/'+final_model+'.h5')


#dev
pred  = model.predict([dev_data[data.continuous_black], dev_mfcc, dev_feats_p, dev_feats_w]).flatten()
data.test_wer (pred, dev_data, "dev_mgb2_"+final_model)

#test
pred  = model.predict([test_data[data.continuous_black], test_mfcc, test_feats_p, test_feats_w]).flatten()
data.test_wer (pred, test_data,  "summa_"+final_model)
#### 

