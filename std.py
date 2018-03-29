import numpy as np
def std_dev (input_file):
    std = np.std(np.loadtxt(input_file,usecols=range(0,1),dtype='float32'))
    return std

std_ref1   = std_dev ("wer_ref.1")
std_glass1 = std_dev ("wer_glass.1")
std_black1 = std_dev ("wer_black.1")

print "standard deviation on WER dev data"
print "ref:", std_ref1, " / glass:", std_glass1, " / black:", std_black1

std_ref2   = std_dev ("wer_ref.2")
std_glass2 = std_dev ("wer_glass.2")
std_black2 = std_dev ("wer_black.2")

print "standard deviation on WER test data"
print "ref:", std_ref2, " / glass:", std_glass2, " / black:", std_black2

