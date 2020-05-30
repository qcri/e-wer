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

## Citation

This data and the reported results are described in [this](http://aclweb.org/anthology/P18-2004) paper:

```bib
@InProceedings{,
    author={Ali, Ahmed and Renals, Steve},
      title={Word Error Rate Estimation for Speech Recognition: e-WER},
      booktitle={ACL},
      year={2018}, 
}
