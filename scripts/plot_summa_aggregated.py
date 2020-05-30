import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

# Copyright 2020  QCRI, HBKU (Author: Ahmed Ali)
# Apache 2.0.


wer_glass= np.load("./results/summa_eWER_glass_box_pred_agg.npy")
wer_black= np.load("./results/summa_eWER_black_box_pred_agg.npy")


wer2_glass= np.load("./results/summa_eWER2_black_box_dnn_pred_agg.npy")
wer2_black= np.load("./results/summa_eWER2_glass_box_dnn_pred_agg.npy")
wer2_no= np.load("./results/summa_eWER2_no_box_dnn_pred_agg.npy")

ref= np.load("./results/summa_eWER2_glass_box_cnn_ref_agg.npy")


plt.clf()
       
x=(np.arange(1,ref.size+1))

ax = plt.subplot(111)
 
plt.plot(x, ref, color='r',label='reference')

plt.plot(x, wer_glass,color='m',label='e-WER glass-box')
plt.plot(x, wer_black, color='b',label='e-WER black-box')


plt.plot(x, wer2_glass,color='m',label='e-WER2 glass-box')
plt.plot(x, wer2_black, color='b',label='e-WER2 black-box')
plt.plot(x, wer2_no,color='k',label='e-WER2 no-box')
    
plt.title("SUMMA Test Aggregated WER")
legend()
plt.savefig("plot_summa_aggregated_ewer.pdf")


