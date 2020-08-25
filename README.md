# Word Error Rate Estimation Without ASR Output: e-WER2

  This is the second version of e-WER (e-WER2).

# New Features!

- An end-to-end multistream architecture to predictthe WER per sentence using language-independent phonotactic features.
- Our novel system is able to learn acoustic-lexical embeddings 
- We estimate the error rate directly without having access to the ASR results nor the ASR system – *no-box* WER estimation 


| System | Pearson | RSME | e-WER (ref WER=28.5) | 
| ------ | ------ |  ------ | ------ |
| e-WER Glass Box | 0.82 | 0.17 | 27.3% |
| e-WER Black Box | 0.68 | 0.19 | 35.8% |
| e-WER2 Glass Box | 0.74 |  0.19.| 27.9% |
| e-WER2 Black Box | 0.66 |  0.21 | 30.9% | 
| e-WER No Box| 0.56 | 0.24 | 30.9% |

# Model definition
An end-to-end multistream based regression model to predict the WER per sentence.

We combine the four streams: lexical, phonotactic, acoustics and numerical features into a single end-to-end network to estimate word error rate directly. We jointly train the multistream network and their final hidden layers are concatenated to obtain a joint feature space in which another fully connected layer to estimate the WER directly. 

<img align="center" width="400" src="https://github.com/qcri/e-wer/blob/e-wer2/images/ewer2.png">

# Results
Test set cumulative WER over all sentences X-axis is duration in hours and Y-axis is WER in %.

<img align="center" width="400" src="https://github.com/qcri/e-wer/blob/e-wer2/images/summa_results.png">

<img align="center" width="400" src="https://github.com/qcri/e-wer/blob/e-wer2/images/mgb2_results.png">


## Citation

This data and the reported results are described in [INTERSPEECH 2020](https://arxiv.org/pdf/2008.03403.pdf) and [ACL 2018](http://aclweb.org/anthology/P18-2004) papers:

```bib
@InProceedings{,
    author={Ali, Ahmed and Renals, Steve},
      title={Word Error Rate Estimation Without ASR Output: e-WER2},
      booktitle={INTERSPEECH},
      year={2020}, 

 @InProceedings{,
    author={Ali, Ahmed and Renals, Steve},
      title={Word Error Rate Estimation for Speech Recognition: e-WER},
      booktitle={ACL},
      year={2018}, 
