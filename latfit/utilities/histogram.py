#!/usr/bin/python3
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.backends.backend_pdf import PdfPages


def main():
    """Make the histograms.
    """
    for fname in sys.argv[1:]:
        make_hist(fname)

def make_hist(fname):
    """
    """
    with open(fname, 'rb') as fn1:
        dat = pickle.load(fn1) 
        dat = np.array(dat)
    avg, err, freqarr = dat
    print(freqarr.shape, avg)
    title = re.sub('_', " ", fname)
    title = re.sub('.p', '', title)
    title = re.sub('0', '', title)
    title = re.sub('mom', 'p', title)
    title = re.sub('x', 'Energy', title)
    for dim in range(freqarr.shape[-1]):
        save_str = re.sub('.p', '_state'+str(dim)+'.pdf', fname)
        with PdfPages(save_str) as pdf:
            title_dim = title+' state:'+str(dim)
            freq = freqarr[:, dim]
            # print(freq, np.mean(freq))
            hist, bins = np.histogram(freq, bins=10)
            # print(hist)
            center = (bins[:-1] + bins[1:]) / 2
            width = 0.7 * (bins[1] - bins[0])
            plt.ylabel('count')
            plt.title(title_dim)
            plt.bar(center, hist, align='center', width=width)
            #fig = plt.gcf()
            print("saving plot as filename:", save_str)
            pdf.savefig()
            plt.show()


if __name__ == '__main__':
    main()
