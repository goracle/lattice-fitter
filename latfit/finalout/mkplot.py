from __future__ import division
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
import os.path
import os
import re

from latfit.config import fit_func
from latfit.config import FINE
from latfit.config import TITLE
from latfit.config import XLABEL
from latfit.config import YLABEL
from latfit.config import UNCORR
from latfit.config import FIT
from latfit.config import METHOD
from latfit.config import BINDS
from latfit.config import START_PARAMS
from latfit.config import AUTO_FIT as auf
from latfit.config import EFF_MASS
from latfit.config import ASSISTED_FIT as asf
from matplotlib import rcParams
import matplotlib.patches as patches
rcParams.update({'figure.autolayout': True})

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
    title=re.sub('_',' ',title)

    #setup fonts
    hfontT = {'fontname':'FreeSans','size':12}
    hfontl = {'fontname':'FreeSans','size':14}

    print "list of plotted points [x,y]:"
    print coords
    XCOORD = [coords[i][0] for i in range(len(coords))]
    YCOORD = [coords[i][1] for i in range(len(coords))]
    ER2 = np.array([np.sqrt(cov[i][i]) for i in range(len(coords))])

    print "list of point errors (x,yerr):"
    print zip(XCOORD,ER2)
    #print message up here because of weirdness with pdfpages
    if FIT:
        if result_min.status == 0:
            sp=np.array(START_PARAMS)
            print "Minimizer thinks that it worked.  Plotting fit."
            print "Fit info:"
            print "Autofit:",auf
            print "Assisted Fit:",asf
            print "Minimizer (of chi^2) method:",METHOD
            if METHOD == 'L-BFGS-B':
                print "Bounds:",BINDS
            print "Guessed params:  ",np.array2string(sp,separator=', ')
            print "Minimized params:",np.array2string(result_min.x, separator=', ')
            print "Error in params :",np.array2string(np.array(param_err), separator=', ')
            print "chi^2 minimized = ", result_min.fun
            dof = len(cov)-len(result_min.x)
            print "degrees of freedom = ", dof
            print "chi^2 reduced = ", result_min.fun/dof

    if EFF_MASS:
        eff_str="_eff_mass"
    else:
        eff_str=""
    with PdfPages(re.sub(' ','_',title)+eff_str+'.pdf') as pdf:
        plt.errorbar(XCOORD, YCOORD, yerr=ER2, linestyle='None',ms=3.75,marker='o')
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
                plt.plot(XFIT, YFIT)
            if len(result_min.x) > 1:
                estring=str(result_min.x[1])+"+/-"+str(param_err[1])
            else:
                #for an effective mass plot
                estring=str(result_min.x[0])+"+/-"+str(param_err[0])
            plt.annotate("Energy="+estring,xy=(0.05,0.95),xycoords='axes fraction')
            dof = len(cov)-len(result_min.x)
            redchisq=result_min.fun/dof
            if redchisq<2:
                plt.annotate("Reduced "+r"$\chi^2=$"+str(redchisq)+",dof="+str(dof),xy=(0.05,0.05),xycoords='axes fraction')
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
            plt.text(XCOORD[3], YCOORD[2],"Uncorrelated fit.")
        #STRIKE2 = "Amplitude = " + str(result_min.x[0]) + "+/-" + str(
        #    ERR_A0)
        #X_POS_OF_FIT_RESULTS = XCOORD[3]
        #plt.text(X_POS_OF_FIT_RESULTS, YCOORD[3], STRIKE1)
        #plt.text(X_POS_OF_FIT_RESULTS, YCOORD[7], STRIKE2)
        plt.xlabel(XLABEL,**hfontl)
        plt.ylabel(YLABEL,**hfontl)
        #plot box for effective mass
        if EFF_MASS and FIT:
            ax = plt.gca()
            #gca,gcf=getcurrentaxes getcurrentfigure
            fig = plt.gcf()
            ax.add_patch((
                plt.Rectangle(#(11.0, 0.24514532441),3,.001,
                    (XCOORD[0]-1, result_min.x[0]-param_err[0]),   # (x,y)
                    XCOORD[len(XCOORD)-1]-XCOORD[0]+2, # width
                    2*param_err[0],          # height
                    fill=True,color='k',alpha=0.5,zorder=1000,figure=fig,
                    #transform=fig.transFigure
                )))
        #read out into a pdf
        pdf.savefig()
        #show the plot
        plt.show()
    return 0
