#!/usr/bin/python3
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import pickle
import operator
from matplotlib.backends.backend_pdf import PdfPages


def main():
    """Make the histograms.
    """
    for fname in sys.argv[1:]:
        make_hist(fname)

def make_hist(fname):
    """Make histograms
    """
    with open(fname, 'rb') as fn1:
        dat = pickle.load(fn1)
        dat = np.array(dat)
        avg, err, freqarr = dat
    spl = fname.split('_')[0]

    # get file name for pvalue
    pvalfn = re.sub(spl, 'pvalue', fname)
    pvalfn = re.sub('shift_', '', pvalfn) # todo: make more general

    with open(pvalfn, 'rb') as fn1:
        pdat = pickle.load(fn1)
        pdat = np.array(pdat)
        pdat_avg, pdat_err, pdat_freqarr = pdat

    # get file name for error
    errfn = re.sub('_mom', '_err_mom', fname)
    with open(errfn, 'rb') as fn1:
        errdat = pickle.load(fn1)
        errdat = np.array(errdat)

    print(freqarr.shape, avg)

    title = gettitle(fname)

    for dim in range(freqarr.shape[-1]):
        save_str = re.sub('.p', '_state'+str(dim)+'.pdf', fname)
        with PdfPages(save_str) as pdf:
            title_dim = title+' state:'+str(dim)
            freq = freqarr[:, dim]
            print("val; pvalue; err")
            pdat_median = np.median(pdat_freqarr)
            median_diff = np.inf
            for i, j, k in sorted(zip(freq, pdat_freqarr, errdat[
                    :, dim]), key = operator.itemgetter(0)):
                if abs(j - pdat_median) <= median_diff:
                    median_diff = abs(j-pdat_median)
                    freq_median = i
                print(np.real(i), j, np.real(k))
            sys_err = np.std(freq, ddof=1)
            # print(freq, np.mean(freq))
            hist, bins = np.histogram(freq, bins=10)
            # print(hist)
            center = (bins[:-1] + bins[1:]) / 2
            width = 0.7 * (bins[1] - bins[0])
            plt.ylabel('count')
            plt.title(title_dim)
            xerr = np.array(errdat[:, dim], dtype=np.float)
            xerr = getxerr(freq, center, errdat[:, dim])
            assert not isinstance(xerr[0], str), "xerr needs conversion"
            assert isinstance(xerr[0], float), "xerr needs conversion"
            assert isinstance(xerr[0], np.float), "xerr needs conversion"
            plt.bar(center, hist, xerr=xerr, align='center', width=width)
            plt.annotate("median="+str(freq_median), xy=(0.05, 0.8),
                         xycoords='axes fraction')
            plt.annotate("standard dev (est of systematic)="+str(sys_err),
                         xy=(0.05, 0.7),
                         xycoords='axes fraction')
            #fig = plt.gcf()
            print("saving plot as filename:", save_str)
            print("xerr =", xerr)
            pdf.savefig()
            plt.show()

def getxerr(freq, center, errdat_dim):
    """Get horiz. error bars"""
    err = np.zeros(len(center), dtype=np.float)
    for n, cent in enumerate(center):
        mindiff = np.inf 
        flag = False
        for k, pair in enumerate(zip(freq, errdat_dim)):
            i, j = pair
            mindiff = min(abs(cent-i), abs(mindiff))
            if mindiff == abs(cent-i):
                flag = True
                err[n] = j
        assert flag, "bug"
    return err

                
            


def gettitle(fname):
    """Get title for histograms"""
    # get title
    title = re.sub('_', " ", fname)
    title = re.sub('.p', '', title)
    title = re.sub('0', '', title)
    title = re.sub('mom', 'p', title)
    title = re.sub('x', 'Energy (lattice units)', title)
    return title


if __name__ == '__main__':
    main()
