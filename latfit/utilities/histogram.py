#!/usr/bin/python3
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import pickle
import operator
from matplotlib.backends.backend_pdf import PdfPages
import gvar


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
        dat = np.real(dat)
        avg, err, freqarr = dat
        avg = gvar.gvar(avg, err)
    spl = fname.split('_')[0]

    # get file name for pvalue
    pvalfn = re.sub(spl, 'pvalue', fname)
    pvalfn = re.sub('shift_', '', pvalfn) # todo: make more general

    with open(pvalfn, 'rb') as fn1:
        pdat = pickle.load(fn1)
        pdat = np.array(pdat)
        pdat = np.real(pdat)
        pdat_avg, pdat_err, pdat_freqarr = pdat

    # get file name for error
    if 'mom' not in fname:
        errfn = fname.replace('_', "_err_", 1)
    else:
        errfn = re.sub('_mom', '_err_mom', fname)
    with open(errfn, 'rb') as fn1:
        errdat = pickle.load(fn1)
        errdat = np.real(np.array(errdat))
    assert len(errdat) > 0, "error array not found"

    print('shape:', freqarr.shape, avg)
    print('shape2:', errdat.shape)

    title = gettitle(fname)

    for dim in range(freqarr.shape[-1]):
        save_str = re.sub(r'.p$', '_state'+str(dim)+'.pdf', fname)
        with PdfPages(save_str) as pdf:
            title_dim = title+' state:'+str(dim)
            freq = np.array([np.real(i) for i in freqarr[:, dim]])
            print("val(err); pvalue; median difference; avg difference")
            pdat_median = np.median(pdat_freqarr)
            median_diff = np.inf
            median_diff2 = np.inf
            half = 0
            errlooparr = errdat[:, dim] if len(errdat.shape) > 1 else errdat
            loop = sorted(zip(freq, pdat_freqarr, errlooparr),
                          key = lambda elem: elem[2], reverse=True)
            median_err = []
            for i, j, k in loop:
                if abs(j - pdat_median) <= median_diff:
                    median_diff = abs(j-pdat_median)
                    freq_median = i
                elif abs(j - pdat_median) <= median_diff2:
                    median_diff2 = abs(j-pdat_median)
                    half = i
                median_err.append([gvar.gvar(np.real(i), np.real(k)), j])
                #print(median_err[-1], j)
            if median_diff != 0:
                freq_median = (freq_median+half)/2
            try:
                sys_err = np.std(np.array(median_err), ddof=1)
            except ZeroDivisionError:
                print("zero division error:")
                print(np.array(median_err))
                sys.exit(1)
            # print(freq, np.mean(freq))
            hist, bins = np.histogram(freq, bins=10)
            # print(hist)
            center = (bins[:-1] + bins[1:]) / 2
            width = 0.7 * (bins[1] - bins[0])
            erronerrmedianstr = str(gvar.gvar(freq_median, sys_err.sdev)).split('(')[1]
            median = gvar.gvar(freq_median, sys_err.val)
            for i,pval in median_err:
                median_diff = i-median
                median_diff = gvar.gvar(median_diff.val, max(i.sdev, median.sdev))
                avg_diff = i-avg[dim]
                avg_diff = gvar.gvar(avg_diff.val, max(i.sdev, avg[dim].sdev))
                if abs(avg_diff.val)>abs(avg_diff.sdev) or abs(median_diff.val)>abs(median_diff.sdev):
                    print(i, pval, median_diff, avg_diff)
                else:
                    print(i, pval)
            print('p-value weighted median =', str(median))
            print("p-value weighted mean =", avg[dim])
            plt.ylabel('count')
            plt.title(title_dim)
            xerr = np.array(errlooparr, dtype=np.float)
            xerr = getxerr(freq, center, errlooparr)
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
