from keras.layers import Input
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Embedding


# Copyright 2020  QCRI, HBKU (Author: Ahmed Ali)
# Apache 2.0.



def create_mlp_ewer (dim):
    inputs   = Input(shape=(dim,))
    fc1      = Dense(128, activation='relu')(inputs)
    dropout1 = Dropout(0.2) (fc1)
    fc2      = Dense(64, activation='relu')(dropout1)
    dropout2 = Dropout(0.2) (fc2)


def wer_regression_DNN_model(inFeat,featLen,labels,outModel,_EPOCHS=10):
    
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=featLen))
    model.add(Dropout(0.2))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    plot_model(model, to_file='dnn_ewer2.pdf', show_shapes=True, show_layer_names=True)
    model.fit(inFeat,labels, epochs=_EPOCHS,batch_size=32,verbose=1)
    model.save(outModel, overwrite=True)  

def create_mlp(dim):
    inputs   = Input(shape=(dim,))
    fc1      = Dense(128, activation='relu')(inputs)
    dropout1 = Dropout(0.2) (fc1)
    fc2      = Dense(64, activation='relu')(dropout1)
    dropout2 = Dropout(0.2) (fc2)

    # construct the mlp
    model = Model(inputs, dropout2)

    return model

def create_1d_txt_cnn(sequence_length=100,vocabulary_size=12309,embedding_dim=256):

  max_features = vocabulary_size
  maxlen = sequence_length
  embedding_dims = 256
  filters = 250
  kernel_size = 3
  hidden_dims = 128
  
  inputs = Input(shape=(sequence_length,), dtype='int32')
  embeds = Embedding(max_features,embedding_dim,input_length=maxlen) (inputs)
  dropout1 = Dropout(0.2) (embeds)

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
  cov1 = Conv1D (filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1) (dropout1)

# we use max pooling:
  maxpool = GlobalMaxPooling1D() (cov1)

# We add a vanilla hidden layer:
  fc1 = Dense(128,activation='relu') (maxpool)
  dropout2 = Dropout(0.2) (fc1)
  fc2 = Dense(64,activation='relu') (dropout2)
  dropout3 = Dropout(0.2) (fc2)

  # construct the CNN 
  model = Model(inputs, dropout3)

  return model
  
def create_3d_txt_cnn(sequence_length=100,vocabulary_size=12309,embedding_dim=256):

  max_features = vocabulary_size
  maxlen = sequence_length
  drop = 0.2
  filter_sizes = [3,4,5]
  num_filters = 512
  
  inputs = Input(shape=(sequence_length,), dtype='int32')
  embedding  = Embedding(max_features,embedding_dim,input_length=maxlen) (inputs)
  reshape = Reshape((sequence_length,embedding_dim,1))(embedding)
  #dropout1 = Dropout(0.2) (embedding)

  conv_0 = Convolution2D(num_filters, filter_sizes[0], embedding_dim,
                       border_mode='valid', init='normal', 
                       activation='relu', dim_ordering='tf')(reshape)
  conv_1 = Convolution2D(num_filters, filter_sizes[1], embedding_dim,
                       border_mode='valid', init='normal',
                       activation='relu', dim_ordering='tf')(reshape)
  conv_2 = Convolution2D(num_filters, filter_sizes[2], embedding_dim,
                       border_mode='valid', init='normal',
                       activation='relu', dim_ordering='tf')(reshape)

  maxpool_0 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1),
                         strides=(1,1), border_mode='valid',
                         dim_ordering='tf')(conv_0)
  maxpool_1 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), 
                         strides=(1,1), border_mode='valid', 
                         dim_ordering='tf')(conv_1)
  maxpool_2 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1),
                         strides=(1,1), border_mode='valid',
                         dim_ordering='tf')(conv_2)

  #merged_tensor = Merge([maxpool_0, maxpool_1, maxpool_2], mode='concat', concat_axis=1)
  merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2])
  flatten = Flatten()(merged_tensor)
  dropout = Dropout(drop)(flatten)

# We add a vanilla hidden layer:
  fc1 = Dense(128,activation='relu') (dropout)
  dropout2 = Dropout(drop) (fc1)
  fc2 = Dense(64,activation='relu') (dropout2)
  dropout3 = Dropout(drop) (fc2)

  # construct the CNN 
  model = Model(inputs, dropout3)

  return model

def create_acoustics_cnn():
    dropout_rate = 0.2
    # Accoustics MFCC features model
    acoustics_input = Input(shape=(2000, 13), name='mfcc_input')

    acoustics_model = Dropout(dropout_rate)(acoustics_input)
    acoustics_model = Conv1D(500,
                             5,
                             padding='same',
                             activation='relu',
                             strides=1)(acoustics_model)
    acoustics_model = Dropout(dropout_rate)(acoustics_model)
    acoustics_model = Conv1D(500,
                             7,
                             padding='same',
                             activation='relu',
                             strides=2)(acoustics_model)
    acoustics_model = Dropout(dropout_rate)(acoustics_model)
    acoustics_model = Conv1D(500,
                             1,
                             padding='same',
                             activation='relu',
                             strides=2)(acoustics_model)
    acoustics_model = Dropout(dropout_rate)(acoustics_model)
    acoustics_model = Conv1D(500,
                             1,
                             padding='same',
                             activation='relu',
                             strides=1)(acoustics_model)
    acoustics_model = Dropout(dropout_rate)(acoustics_model)

    acoustics_model = GlobalMaxPooling1D()(acoustics_model)

    acoustics_model = Dropout(dropout_rate)(acoustics_model)
    acoustics_model = Dense(500, activation='softmax')(acoustics_model)
    
    acoustics_model = Dense(64,activation='relu') (acoustics_model)
    acoustics_model = Dropout(dropout_rate) (acoustics_model)

    acoustics_model = Model(inputs=acoustics_input, outputs=acoustics_model)
    
    return acoustics_model