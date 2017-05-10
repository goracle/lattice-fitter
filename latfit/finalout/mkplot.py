from __future__ import division
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
import os.path
import os

from latfit.config import fit_func
from latfit.config import FINE
from latfit.config import TITLE
from latfit.config import XLABEL
from latfit.config import YLABEL
from latfit.config import UNCORR
from latfit.config import FIT

def mkplot(coords, cov, INPUT,result_min=None, param_err=None):
    """Plot the fitted graph."""
    if TITLE == '' or not TITLE:
        #then plot title should be the location directory of the jk blocks
        cwd = os.getcwd()
        if os.path.isdir(INPUT):
            os.chdir(INPUT)
            title=os.getcwd().split('/')[-1]
            os.chdir(cwd)
        else:
            title=INPUT
    else:
        title=TITLE

    #setup fonts
    hfontT = {'fontname':'FreeSans','size':12}
    hfontl = {'fontname':'FreeSans','size':14}

    with PdfPages(title+'.pdf') as pdf:
        XCOORD = [coords[i][0] for i in range(len(coords))]
        YCOORD = [coords[i][1] for i in range(len(coords))]
        ER2 = np.array([np.sqrt(cov[i][i]) for i in range(len(coords))])
        plt.errorbar(XCOORD, YCOORD, yerr=ER2, linestyle='None',ms=0.75,marker='.')
        print "(y)Error bar numeric values:",ER2
        if FIT:
            #the fit function is plotted on a scale FINE times more fine
            #than the original data points
            step = abs((XCOORD[len(XCOORD)-1]-XCOORD[0]))/FINE/(len(XCOORD)-1)
            XFIT = np.arange(XCOORD[0], XCOORD[len(XCOORD)-1]+step,step)
            #result_min.x is is the array of minimized fit params
            YFIT = np.array([fit_func(XFIT[i], result_min.x)
                             for i in range(len(XFIT))])
            #only plot fit function if minimizer result makes sense
            if result_min.status == 0:
                print "Minimizer thinks that it worked.  Plotting fit."
                plt.plot(XFIT, YFIT)
            plt.text(XCOORD[1], YCOORD[0],"Energy="+str(result_min.x[1])+"+/-"+str(param_err[1]))
        #todo: figure out a way to generally assign limits to plot
        #plt.xlim([XCOORD[0], XMAX+1])
        #magic numbers for the problem you're solving
        #plt.ylim([0, 0.1])
        #add labels, more magic numbers
        plt.title(title,**hfontT)
        #todo: figure out a way to generally place text on plot
        #STRIKE1 = "Energy = " + str(result_min.x[1]) + "+/-" + str(
        #    ERR_ENERGY)
        if UNCORR:
            plt.text(XCOORD[3], YCOORD[1],"Uncorrelated fit.")
        #STRIKE2 = "Amplitude = " + str(result_min.x[0]) + "+/-" + str(
        #    ERR_A0)
        #X_POS_OF_FIT_RESULTS = XCOORD[3]
        #plt.text(X_POS_OF_FIT_RESULTS, YCOORD[3], STRIKE1)
        #plt.text(X_POS_OF_FIT_RESULTS, YCOORD[7], STRIKE2)
        plt.xlabel(XLABEL,**hfontl)
        plt.ylabel(YLABEL,**hfontl)
        #read out into a pdf
        pdf.savefig()
        #show the plot
        plt.show()
    return 0
