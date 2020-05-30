import data
import models
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


_feature="grapheme"
_ngram=3
word_list_g = data.return_word_list (train_data[_feature].values.tolist(),_ngram)
train_feats_g = data.make_features (train_data[_feature].values.tolist(), word_list_g, _ngram, True)
dev_feats_g = data.make_features (dev_data[_feature].values.tolist(), word_list_g, _ngram, True)
test_feats_g = data.make_features (test_data[_feature].values.tolist(), word_list_g, _ngram,  True)


_feature="words"
_ngram=2
word_list_w = data.return_word_list (train_data[_feature].values.tolist(), _ngram)
train_feats_w = data.make_features (train_data[_feature].values.tolist(), word_list_w, _ngram, True)
dev_feats_w = data.make_features (dev_data[_feature].values.tolist(), word_list_w, _ngram, True)
test_feats_w = data.make_features (test_data[_feature].values.tolist(), word_list_w, _ngram, True)


train_feats = np.hstack((train_feats_g,train_feats_w,train_data[data.continuous_glass_grapheme]))
dev_feats   = np.hstack((dev_feats_g,dev_feats_w,dev_data[data.continuous_glass_grapheme]))
test_feats  = np.hstack((test_feats_g,test_feats_w,test_data[data.continuous_glass_grapheme]))


print (train_feats_g.shape,train_feats_w.shape,train_data[data.continuous_glass_grapheme].shape,train_feats.shape)


## call back 
mlp = models.create_mlp(train_feats.shape[1])


wer = Dense(32, activation="relu")(mlp.output)
wer = Dropout(0.2) (wer)
wer = Dense(1, kernel_initializer='normal')(wer)


##
final_model='eWER_glass_box'

model = Model(inputs=mlp.input, outputs= wer)
plot_model(model, to_file='results/'+final_model+'_plots.pdf', show_shapes=True, show_layer_names=True)
model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])


# TRAIN E-WER
'''
earlystopper = EarlyStopping(monitor='val_loss', min_delta=0,
                             patience=1, verbose=1, mode='auto')

checkpoint = ModelCheckpoint(filepath=final_model+'.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_accuracy', 
                             verbose=1, save_best_only=True, mode='auto')


model.fit(train_feats, train_data["wer"].to_numpy(),
           validation_data=(dev_feats, dev_data["wer"].to_numpy()),
           batch_size=32,  epochs=_EPOCHS, verbose=1, 
           callbacks=[checkpoint,earlystopper])  
           
'''
model.summary()

           

model.fit(train_feats, train_data["wer"].to_numpy(),
           batch_size=32,  epochs=_EPOCHS, verbose=1)

model.save('results/'+final_model+'.h5', overwrite=True)
model = load_model('results/'+final_model+'.h5')
print (dev_feats.shape[1])


#dev
pred  = model.predict(dev_feats).flatten()
data.test_wer (pred, dev_data, "dev_mgb2_"+final_model)

#test
pred  = model.predict(test_feats).flatten()
data.test_wer (pred, test_data,  "summa_"+final_model)
#### 

