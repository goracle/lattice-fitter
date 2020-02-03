#!/usr/bin/python3

import sys
import re
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

fname = sys.argv[1]
print('plotting:', fname)
figx = pickle.load(open(fname, 'rb'))
pname = re.sub('.p', '.pdf', fname)
print('saving plot as:', pname)
pfig = PdfPages(pname)
pfig.savefig(figx)
plt.show() # Show the figure, edit it, etc.!
