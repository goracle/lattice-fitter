"""Make the final plot."""
import os.path
import os
import re
import sys
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
import matplotlib.pyplot as plt

from latfit.config import fit_func
from latfit.config import FINE
from latfit.config import TITLE
from latfit.config import TITLE_PREFIX
from latfit.config import XLABEL
from latfit.config import YLABEL
from latfit.config import UNCORR
from latfit.config import FIT
from latfit.config import METHOD
from latfit.config import BINDS
from latfit.config import START_PARAMS
from latfit.config import AUTO_FIT
from latfit.config import EFF_MASS
from latfit.config import EFF_MASS_METHOD
from latfit.config import C
from latfit.config import NO_PLOT
from latfit.config import ASSISTED_FIT
from latfit.config import GEVP
from latfit.config import JACKKNIFE_FIT
from latfit.config import JACKKNIFE
rcParams.update({'figure.autolayout': True})

def mkplot(coords, cov, input_f, result_min=None, param_err=None):
    """Plot the fitted graph."""

    #get dimension of GEVP, or set to one if not doing gevp (this is needed in several places)
    try:
        dimops = len(cov[0][0])
    except TypeError:
        dimops = 1

    #title/filename stuff
    if TITLE == '' or not TITLE:
        #then plot title should be the location directory of the jk blocks
        cwd = os.getcwd()
        if os.path.isdir(input_f):
            os.chdir(input_f)
            title = os.getcwd().split('/')[-1]
            os.chdir(cwd)
        else:
            title = input_f
    else:
        title = TITLE
    title = TITLE_PREFIX+title
    title = re.sub('_', ' ', title)
    #brief attempt at sanitization
    title_safe = re.sub(r'\$', '', title)
    title_safe = re.sub(r'\\', '', title_safe)
    title_safe = re.sub(r', ', '', title_safe)
    if JACKKNIFE_FIT == 'DOUBLE':
        jk_str = '_2xjk'
    elif JACKKNIFE_FIT == 'FROZEN':
        jk_str = '_1xjk'
    else:
        jk_str = ''
    if EFF_MASS:
        eff_str = '_eff_mass'
        if EFF_MASS_METHOD == 1:
            eff_str += '_meth1'
            print("C = ", C)
    else:
        eff_str = ''
    if UNCORR:
        print("Doing uncorrelated fit.")
        uncorr_str = '_uncorr_fit'
    else:
        uncorr_str = ''
    if GEVP:
        gevp_str = ' GEVP '+str(dimops)+'dim'
    else:
        gevp_str = ''

    #setup fonts
    hfontt = {'fontname':'FreeSans', 'size':12}
    hfontl = {'fontname':'FreeSans', 'size':14}

    #plotted coordinates setup
    print("list of plotted points [x, y]:")
    print(coords)
    xcoord = [coords[i][0] for i in range(len(coords))]
    ycoord = [coords[i][1] for i in range(len(coords))]
    error2 = np.array([np.sqrt(cov[i][i]) for i in range(len(coords))])
    print("list of point errors (x, yerr):")
    print(list(zip(xcoord, error2)))

    #print message up here because of weirdness with pdfpages
    if FIT:
        if result_min.status == 0:
            startp = np.array(START_PARAMS)

            #print plot info
            print("Minimizer thinks that it worked.  Plotting fit.")
            print("Fit info:")
            print("Autofit:", AUTO_FIT)
            print("Assisted Fit:", ASSISTED_FIT)
            print("Minimizer (of chi^2) method:", METHOD)
            if METHOD == 'L-BFGS-B':
                print("Bounds:", BINDS)
            print("Guessed params:  ", np.array2string(startp, separator=', '))
            print("Minimized params:", np.array2string(result_min.x, separator=', '))
            print("Error in params :", np.array2string(np.array(param_err), separator=', '))
            chisq_str = str(result_min.fun)
            if JACKKNIFE_FIT:
                chisq_str += '+/-'+str(result_min.err_in_chisq)
            print("chi^2 minimized = ", chisq_str)
            dof = len(cov)*dimops-len(result_min.x)
            #Do this because C parameter is a fit parameter, it just happens to be guessed by hand
            if EFF_MASS and EFF_MASS_METHOD == 1 and C != 0.0:
                dof -= 1
            print("degrees of freedom = ", dof)
            redchisq = result_min.fun/dof
            redchisq_str = str(redchisq)
            if JACKKNIFE_FIT:
                redchisq_str += '+/-'+str(result_min.err_in_chisq/dof)
                if (redchisq > 10 or redchisq < 0.1) or (
                        result_min.err_in_chisq/dof > 10
                        or result_min.err_in_chisq/dof < .1):
                    redchisq_round_str = '{:0.7e}'.format(redchisq)+'+/-'
                    redchisq_round_str += '{:0.7e}'.format(
                        result_min.err_in_chisq/dof)
                else:
                    redchisq_round_str = '{:0.8}'.format(redchisq)
                    redchisq_round_str += '+/-'+'{:0.8}'.format(
                        result_min.err_in_chisq/dof)
            print("chi^2 reduced = ", redchisq_str)
            dimops_chk = len(fit_func(xcoord[0], result_min.x))
            if dimops != dimops_chk:
                print("***ERROR***")
                print("Fit function length does not match cov. mat.")
                print("Debug of config necessary.")
                print(dimops, dimops_chk)
                sys.exit(1)

    if NO_PLOT:
        return 0
    with PdfPages(re.sub(' ', '_', title_safe+eff_str+uncorr_str+gevp_str+jk_str+'.pdf')) as pdf:
        if dimops != 1:
            lcoord = len(xcoord)
            for curve_num in range(dimops):
                ycurve = np.array([ycoord[i][curve_num] for i in range(lcoord)])
                yerr = np.array([error2[i][curve_num][curve_num] for i in range(lcoord)])
                plt.errorbar(xcoord, ycurve, yerr=yerr,
                             linestyle='None', ms=3.75, marker='o',
                             label='Energy('+str(curve_num)+')')
        else:
            plt.errorbar(xcoord, ycoord, yerr=error2,
                         linestyle='None', ms=3.75, marker='o')
        if FIT:
            #the fit function is plotted on a scale FINE times more fine
            #than the original data points (to show smoothness)
            step_size = abs((xcoord[len(xcoord)-1]-xcoord[0]))/FINE/(
                len(xcoord)-1)
            xfit = np.arange(xcoord[0], xcoord[len(xcoord)-1]+step_size,
                             step_size)
            for curve_num in range(len(fit_func(xfit[0], result_min.x))):
                #result_min.x is is the array of minimized fit params
                yfit = np.array([
                    fit_func(xfit[i], result_min.x)[curve_num]
                    for i in range(len(xfit))])
                #only plot fit function if minimizer result makes sense
                if result_min.status == 0:
                    plt.plot(xfit, yfit)

            #plot tolerance box around straight line fit for effective mass
            if EFF_MASS:
                axvar = plt.gca()
                #gca, gcf = getcurrentaxes getcurrentfigure
                fig = plt.gcf()
                if GEVP:
                    for i in range(dimops):
                        axvar.add_patch((
                            plt.Rectangle(#(11.0, 0.24514532441), 3,.001,
                                (xcoord[0]-1, result_min.x[i]-param_err[i]),   # (x, y)
                                xcoord[len(xcoord)-1]-xcoord[0]+2, # width
                                2*param_err[i],          # height
                                fill=True, color='k', alpha=0.5, zorder=1000, figure=fig,
                                #transform=fig.transFigure
                            )))
                else:
                    axvar.add_patch((
                        plt.Rectangle(#(11.0, 0.24514532441), 3,.001,
                            (xcoord[0]-1, result_min.x[0]-param_err[0]),   # (x, y)
                            xcoord[len(xcoord)-1]-xcoord[0]+2, # width
                            2*param_err[0],          # height
                            fill=True, color='k', alpha=0.5, zorder=1000, figure=fig,
                            #transform=fig.transFigure
                        )))

            #annotate plot with fitted energy
            if GEVP:
                plt.legend(loc='upper right')
                for i, min_e in enumerate(result_min.x):
                    estring = str(min_e)+"+/-"+str(param_err[i])
                    plt.annotate(
                        "Energy["+str(i)+"] = "+estring,
                        xy=(0.05, 0.95-i*.05), xycoords='axes fraction')
            else:
                if len(result_min.x) > 1:
                    estring = str(result_min.x[1])+"+/-"+str(param_err[1])
                else:
                    #for an effective mass plot
                    estring = str(result_min.x[0])+"+/-"+str(param_err[0])
                plt.annotate("Energy="+estring, xy=(0.05, 0.95), xycoords='axes fraction')

            #if the fit is good, also annotate with reduced chi^2
            if result_min.status == 0:
                if redchisq < 2:
                    if EFF_MASS_METHOD == 3:
                        rcp = "Reduced "+r"$\chi^2 = $"
                        rcp += redchisq_round_str+", dof = "+str(dof)
                        plt.annotate(rcp,
                                     xy=(0.05, 0.85),
                                     xycoords='axes fraction')
                    else:
                        plt.annotate(
                            "Reduced "+r"$\chi^2=$"+redchisq_round_str+", dof="+str(dof),
                            xy=(0.05, 0.05),
                            xycoords='axes fraction')
            if UNCORR:
                if dimops > 1:
                    plt.annotate("Uncorrelated fit.", xy=(0.05, 0.10), xycoords='axes fraction')
                else:
                    plt.text(xcoord[3], ycoord[2], "Uncorrelated fit.")
            if JACKKNIFE_FIT:
                if JACKKNIFE_FIT == 'FROZEN' or JACKKNIFE_FIT == 'SINGLE':
                    jk_txt = 'Frozen (single) jackknife fit.'
                elif JACKKNIFE_FIT == 'DOUBLE':
                    jk_txt = 'Double jackknife fit.'
                else:
                    jk_txt = ''
            elif JACKKNIFE:
                jk_txt = 'Avg. fit, jackknife est. cov. matrix'
            else:
                jk_txt = ''
            plt.annotate(jk_txt, xy=(0.05, 0.15), xycoords='axes fraction')

        #add axes labels, title
        plt.title(title, **hfontt)
        plt.xlabel(XLABEL, **hfontl)
        plt.ylabel(YLABEL, **hfontl)

        #read out into a pdf
        pdf.savefig()
        #show the plot
        plt.show()
    return 0
