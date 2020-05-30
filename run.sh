#!/bin/bash

# Copyright 2020  QCRI, HBKU (Author: Ahmed Ali)
# Apache 2.0.

#1) downlaod wav and MFCC data (needed for e-WER2 scripts: step5-7) 
wget http://crowdsource.cloudapp.net/e-wer2/data/wav_splits.tar
tar -xvf wav_splits.tar
mv wav_splits data 


#2) extract mfcc 
for dir in data/wav_splits/*; do
	python scripts/essentia_mfcc.py $dir
	python scripts/compute_mean_var_mfcc_no_cvmn_essentia.py $dir
done 

mkdir -p results log 

#3)  e-wer glass box regression 
time python scripts/train_ewer_glass_box.py &> log/ewer_glass_box.log


#4) e-wer black box regression 
time python scripts/train_ewer_black_box.py &> log/ewer_black_box.log 

#5) e-wer2 glass box 
time python scripts/train_ewer2_glass_box_dnn.py &> log/ewer2_glass_box_dnn.log 
time python scripts/train_ewer2_glass_box_cnn.py &> log/ewer2_glass_box_cnn.log 


#6) e-wer2 black box 
time python scripts/train_ewer2_black_box_dnn.py &> log/ewer2_black_box_dnn.log 
time python scripts/train_ewer2_black_box_cnn.py &> log/ewer2_black_box_cnn.log 

#7) e-wer2 no box 
time python scripts/train_ewer2_no_box_dnn.py &> log/ewer2_no_box_dnn.log 
time python scripts/train_ewer2_no_box_cnn.py &> log/ewer2_no_box_cnn.log 


#8) plot results 
python scripts/plot_summa_aggregated.py
python scripts/plot_mgb2_aggregated.py
