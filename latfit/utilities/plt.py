#!/usr/bin/python3
"""Shows a pickled matplotlib plot and saves a static copy as a pdf"""
import sys
import re
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main():
    """main"""
    fname1 = None
    for fname in sys.argv[1:]:
        #fname = sys.argv[1]
        fname1 = fname
        figx = pickle.load(open(fname, 'rb'))
        print('plotting:', fname, figx)
    if len(sys.argv) == 2:
        pname = re.sub(r'.p$', '.pdf', fname1)
        print('saving plot as:', pname)
        pfig = PdfPages(pname)
        pfig.savefig(figx, bbox_inches='tight')
        plt.savefig(pname, format='pdf', dpi=1200)
    plt.show() # Show the figure, edit it, etc.!

if __name__ == '__main__':
    main()
