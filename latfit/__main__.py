#!/usr/bin/env python

"""Fit function to data.
Compute chi^2 and errors.
Plot fit with error bars.
Save result to pdf.
usage note: MAKE SURE YOU SET THE Y LIMITS of your plot by hand!
usage note(2): MAKE SURE as well that you correct the other "magic"
parts of the graph routine
"""

#install pip2
#probably needs to be refactored for python3...
#then sudo pip install numdifftools

from __future__ import division
from collections import namedtuple
import sys
import numpy as np
import os
from math import sqrt

from latfit.singlefit import singlefit
from latfit.config import JACKKNIFE
from latfit.procargs import procargs
from latfit.extract.errcheck.xlim_err import xlim_err
from latfit.extract.errcheck.xstep_err import xstep_err
from latfit.extract.errcheck.trials_err import trials_err
from latfit.extract.proc_folder import proc_folder
from latfit.finalout.printerr import printerr
from latfit.finalout.mkplot import mkplot

def main():
    ####set up 1ab
    OPTIONS = namedtuple('ops', ['xmin', 'xmax', 'xstep', 'trials'])

    
    ###error processing, parameter extractions
    INPUT, OPTIONS = procargs(sys.argv[1:])
    XMIN, XMAX = xlim_err(OPTIONS.xmin, OPTIONS.xmax)
    XSTEP = xstep_err(OPTIONS.xstep, INPUT)
    TRIALS = trials_err(OPTIONS.trials)

    if TRIALS == -1:
        RESULT_MIN, PARAM_ERR, COORDS, COV = singlefit(INPUT, 
                                                       XMIN, XMAX, XSTEP)
        printerr(RESULT_MIN.x, PARAM_ERR)
        mkplot(COORDS, COV, RESULT_MIN)
        sys.exit(0)
    else:
        list_fit_params = []
        for ctime in range(TRIALS):
           IFILE = proc_folder(INPUT, ctime, "blk")
           NINPUT = os.path.join(INPUT, IFILE)
           RESULT_MIN, PARAM_ERR, COORDS, COV = singlefit(NINPUT, 
                                                          XMIN, XMAX, XSTEP)
           list_fit_params.append(RESULT_MIN.x)
        #todo: make plot section
        TRANSPOSE = np.array(list_fit_params).T.tolist()
        avg_fit_params = [sum(TRANSPOSE[i])/len(TRANSPOSE[i]) for i in range(
        len(TRANSPOSE))]
        if JACKKNIFE == "YES":
            prefactor = (TRIALS-1.0)/(1.0*TRIALS)
        elif JACKKNIFE == "NO":
            prefactor = (1.0)/((TRIALS-1.0)*(1.0*TRIALS))
        else:
            print "***ERROR***"
            print "JACKKNIFE value should be a string with value either"
            print "YES or NO"
            print "Please examine the config file."
            sys.exit(1)
        err_fit_params = [sqrt(sum([(TRANSPOSE[i][j]-avg_fit_params[i])**2
                                    for j in range(len(
                                            TRANSPOSE[i]))])*prefactor)
                          for i in range(len(TRANSPOSE))]
        printerr(avg_fit_params, err_fit_params)
        sys.exit(0)

if __name__ == "__main__":
    main()
