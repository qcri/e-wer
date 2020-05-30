
import numpy as np
import pandas as pd
from nltk import ngrams
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
import re


# Copyright 2020  QCRI, HBKU (Author: Ahmed Ali)
# Apache 2.0.


##

#Features extracted from running grapheme recognition and calucualte grapheme error rate.
#you so this by spittig words to char sequence and align it with the grapheme sequence
grapheme_features  =  ["graphemecount", "gwer", "gerr", "gtotal", "gins", "gdel", "gsub"]

#balc box feature if you have no access to the decoder.
#framecount is debatable, but it is redundant since you can estimate it from duration
black_box_features =  ["wordcount", "duration"]

#this is assumig you have access to the decoder
glass_box_features =  ["avg_loglike", "total_AMloglike", "total_LMloglike"]


###
#Here we list numerical features
###

#BlackBox numerical features 
continuous_black = black_box_features[:]
continuous_no_box = ["duration","phonemeCount"]

#BlackBox numerical features with grapheme error rate
continuous_black_grapheme = black_box_features + grapheme_features

#GlassBox numerical features 
continuous_glass = glass_box_features + black_box_features

#BlackBox numerical features with grapheme error rate
continuous_glass_grapheme = continuous_glass + grapheme_features

# Defining all numerical features. It is listed here here again to maintian the order in the feature file
continuous = []
continuous.extend (["wordcount", "graphemecount", "total_frames", "avg_loglike"])
continuous.extend (["total_AMloglike", "total_LMloglike", "duration"])
continuous.extend (["gwer", "gerr", "gtotal", "gins", "gdel", "gsub"])


cols = [] 
cols.extend (["id", "words", "grapheme", "phoneme"])
cols.extend (continuous)
cols.extend (["wer", "err", "total", "ins", "del", "sub"])


def load_features(feat_file,cols):
  df = pd.read_csv(feat_file, sep="\t", header=None, names=cols)
  
  _list = []
  for x in df["phoneme"].tolist(): 
    _list.append(len(x.split()))
  df["phonemeCount"] = _list
  return df


def word_grams(words, min=1, max=4):
    s = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))
    return s

def return_word_list (_text,_ngram=4):
    _unigram_word_list = [] 
    for i,j in enumerate(_text):
        _words_1gram = word_grams (_text[i].split(),1,_ngram)
        for uni in _words_1gram: 
             if uni not in _unigram_word_list: _unigram_word_list.append(uni)
    return _unigram_word_list
                    
def myvectorize (words,word_list,_ngram):
    feature_vector = []
    index=0
    
    temp_words = [0] * len(word_list)
    for word in word_grams (words.split(),1,_ngram): 
        for index, word2 in enumerate(word_list): 
            if word == word2:
                temp_words[index]+=1
                break
        
    return temp_words

def make_features (text, word_list, _ngram=4, Verbose=False):
    allfeats= []
    for i, _id in enumerate (text):
        feat = myvectorize (text[i],word_list,_ngram)
        allfeats.append (feat)
        if Verbose: 
        	if i % 1000 == 0: print ("Processing: ", i)
    nn=np.array(allfeats)
    if Verbose: print (nn.shape)
    return (nn)


def get_wer (file,dump):
    #print ("Processing: ", file)
    f=open(file,"r")
    lines=f.readlines()
    _wer=[]
    wc_all=0
    err_all=0
    for x in lines:
        err = int(x.split('\t')[18])
        wc  = int(x.split('\t')[19])
        wer = err/wc
        _wer.append (wer)
        wc_all+=wc
        err_all+=err
    return err_all/wc_all, np.array(_wer)


def scaled_wer (wer_list, duration_list):
    scaled_wer = np.sum((wer_list*duration_list))/np.sum(duration_list)*100
    return scaled_wer



def test_wer (pred, test_pd, file_name="test"): 

    ref = test_pd["wer"].to_numpy()
    id = test_pd["id"].to_numpy()
    err = test_pd["err"]
    total = test_pd["total"]
    duration = test_pd["duration"]


    pearsonWER = pearsonr(ref, pred)
    RMSWER = sqrt(mean_squared_error(ref, pred))
    print ("WER: Pearson Correlation: %.2f," % pearsonWER[0], 'RMSE:', "%.2f." %RMSWER)

    ref_wer = err.sum()/total.sum()*100
    ref_scaled_wer = (ref*duration).sum()/duration.sum()*100
    ewer = (pred*duration).sum()/duration.sum()*100
    print ("Ref WER: %.2f," % ref_wer, 'Ref scaled WER:', "%.2f." %ref_scaled_wer, 'e-WER:', "%.2f." %ewer)

    #plot Aggregated WER
    ref_agg = get_acc_wer (ref,duration)
    pred_agg = get_acc_wer (pred,duration)
    
    np.save('results/'+file_name + "_ref_agg", ref_agg)
    np.save('results/'+file_name + "_pred_agg", pred_agg)
    
    plot_line (ref_agg, pred_agg, 'results/'+file_name +"_aggregate.pdf","Aggregated WER")

    #plot Program WER
    ref_program = val_program (test_pd,ref)
    pred_program = val_program (test_pd,pred)
    plot_bar (ref_program, pred_program, 'results/'+file_name + "_Program.pdf", "program WER")
      



def uniq_programs (ids) :
    _unique_id = {}
    for index, content in enumerate (ids):
        
        if "_seg" in content:
            program= re.sub(r'_seg.*', '', content)
        else:
            program= re.sub(r'_\d+\.\d{3}_\d+\.\d{3}', '', content)
        
        if program not in _unique_id.keys(): 
            _unique_id[program] = {}
            _unique_id[program]['start'] = index
        else: 
            _unique_id[program]['end'] = index
    return _unique_id

def val_program (test_pd,wer_np):
    program1 = uniq_programs (test_pd["id"].to_numpy())
    dur_list = test_pd["duration"].to_numpy()
    _dict_count = {}
    _final_list = []
    for program in program1:
        _s = program1[program]['start']
        _e   = program1[program]['end']
        
        sum = scaled_wer (wer_np[_s:_e],dur_list[_s:_e])
        _dict_count [program]=sum
     
    for key in sorted(_dict_count): _final_list.append(_dict_count[key])
    return (np.array(_final_list))


def plot_line (ref,pred,file_to_plot="file_plot.pdf",title="test data") :
    plt.clf()
       
    x=(np.arange(1,ref.size+1))

    ax = plt.subplot(111)
 
    plt.plot(x, ref,color='r',label='reference')
    plt.plot(x, pred,color='b',label='prediction')
    #plt.plot(x, black,color='r',label='black-box')
    

    plt.title(title)
    ax.legend()
    plt.savefig(file_to_plot)

def plot_bar (ref,pred,file_to_plot="file_plot.pdf",title="test data") :
    plt.clf()
       
    x=(np.arange(1,ref.size+1))

    #ax = plt.subplot(111)
 
    #plt.bar(x, ref,color='b',label='reference')
    #plt.bar(x, pred,color='g',label='prediction')

    X = (np.arange(1,ref.size+1))
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(X + 0.25, ref, color = 'r', width = 0.25,label='reference')
    ax.bar(X + 0.00, pred, color = 'b', width = 0.25,label='prediction')
    

    plt.title(title)
    ax.legend()
    plt.savefig(file_to_plot)


def get_acc_wer (wer_list, _dur_list):
    _wer_acc=[]

    for _index, _value in enumerate(wer_list):
        x = scaled_wer (wer_list[0:_index+1],_dur_list[0:_index+1]) 
        _wer_acc.append (x)
    return np.array(_wer_acc)

def create_vectorizer(full_text):
    vectorizer = keras.preprocessing.text.Tokenizer(num_words=None,
                                lower=False,
                                split=" ",
                                char_level=False)
    vectorizer.fit_on_texts(full_text)
    
    return vectorizer

def get_text_vectors(full_text, vectorizer,maxlen=500):
    vector = vectorizer.texts_to_sequences(full_text)
    vector_pad = pad_sequences(vector, maxlen=maxlen)
    
    return vector_pad
