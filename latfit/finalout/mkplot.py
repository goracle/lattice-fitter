from __future__ import division
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt

from latfit.mathfun.fit_func import fit_func

def mkplot(coords, cov, result_min, switch):
    """Plot the fitted graph."""
    with PdfPages('foo.pdf') as pdf:
        XCOORD = [coords[i][0] for i in range(len(coords))]
        YCOORD = [coords[i][1] for i in range(len(coords))]
        ER2 = np.array([cov[i][i] for i in range(len(coords))])
        plt.errorbar(XCOORD, YCOORD, yerr=ER2, linestyle='None')
        #the fit function is plotted on a scale 1000x more fine
        #than the original data points
        XFIT = np.arange(XCOORD[0],
                         XCOORD[len(XCOORD)-1],
                         abs((XCOORD[len(
                             XCOORD)-1]-XCOORD[0]))/1000.0/len(XCOORD))
        YFIT = np.array([fit_func(XFIT[i], result_min.x, switch)
                         for i in range(len(XFIT))])
        #only plot fit function if minimizer result makes sense
        if result_min.status == 0:
            print "Minimizer thinks that it worked.  Plotting fit."
            plt.plot(XFIT, YFIT)
        #todo: figure out a way to generally assign limits to plot
        #plt.xlim([XCOORD[0], XMAX+1])
        #magic numbers for the problem you're solving
        #plt.ylim([0, 0.1])
        #add labels, more magic numbers
        plt.title('Some Correlation function vs. q^2')
        #todo: figure out a way to generally place text on plot
        #STRIKE1 = "Energy = " + str(result_min.x[1]) + "+/-" + str(
        #    ERR_ENERGY)
        #STRIKE2 = "Amplitude = " + str(result_min.x[0]) + "+/-" + str(
        #    ERR_A0)
        #X_POS_OF_FIT_RESULTS = XCOORD[3]
        #plt.text(X_POS_OF_FIT_RESULTS, YCOORD[3], STRIKE1)
        #plt.text(X_POS_OF_FIT_RESULTS, YCOORD[7], STRIKE2)
        plt.xlabel('q^2 (GeV/c^2)^2')
        plt.ylabel('the function')
        #read out into a pdf
        pdf.savefig()
        #show the plot
        plt.show()
    return 0
