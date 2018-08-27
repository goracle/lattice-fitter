"""Make the final plot."""
import os.path
import os
import re
import sys
# from warnings import warn
from decimal import Decimal
from numbers import Number
import itertools
from collections import namedtuple
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy import stats
import h5py

from latfit.utilities import read_file as rf
from latfit.config import fit_func, MOMSTR, L_BOX
from latfit.config import FINE
from latfit.config import TITLE
from latfit.config import XLABEL
from latfit.config import YLABEL
from latfit.config import UNCORR
from latfit.config import FIT
from latfit.config import METHOD
from latfit.config import BINDS
from latfit.config import START_PARAMS
from latfit.config import AUTO_FIT
from latfit.config import EFF_MASS
from latfit.config import BOX_PLOT
from latfit.config import EFF_MASS_METHOD
from latfit.config import C
from latfit.config import NO_PLOT
from latfit.config import ASSISTED_FIT
from latfit.config import GEVP
from latfit.config import JACKKNIFE_FIT
from latfit.config import JACKKNIFE
from latfit.config import PREC_DISP
from latfit.config import STYPE
from latfit.config import ADD_CONST
from latfit.config import ERROR_BAR_METHOD
from latfit.config import CALC_PHASE_SHIFT
from latfit.config import ISOSPIN
from latfit.config import PLOT_DISPERSIVE, DISP_ENERGIES
from latfit.config import AINVERSE
from latfit.config import FIT
from latfit.config import DELTA_E_AROUND_THE_WORLD, MATRIX_SUBTRACTION
from latfit.config import DELTA_E2_AROUND_THE_WORLD
import latfit.config

rcParams.update({'figure.autolayout': True})

NUM_CONFIGS = -1

def update_result_min_nofit(plotdata):
    """Update the result with around the world shift in energy
    associated with non-zero center of mass momentum
    """
    assert MATRIX_SUBTRACTION, "addition of delta E makes sense only if"+\
        " matrix subtraction is being performed"
    for i, _ in enumerate(plotdata.coords):
        plotdata.coords[i][1] += DELTA_E_AROUND_THE_WORLD
        if DELTA_E2_AROUND_THE_WORLD is not None:
            plotdata.coords[i][1] += DELTA_E2_AROUND_THE_WORLD
    return plotdata

def mkplot(plotdata, input_f,
           result_min=None, param_err=None, fitrange=None):
    """Plot the fitted graph."""

    if GEVP:
        plotdata = update_result_min_nofit(plotdata)

    # GET COORDS
    error2 = get_prelim_errbars(result_min)
    if error2 is None:
        print("Using average covariance matrix to find error bars.")
        xcoord, ycoord, error2 = get_coord(plotdata.coords,
                                           plotdata.cov, None)
    else:
        print("Using average error bars from jackknife fit.")
        xcoord, ycoord, error2 = get_coord(plotdata.coords,
                                           plotdata.cov, error2)

    # get dimension of GEVP,
    # or set to one if not doing gevp (this is needed in several places)
    dimops = get_dimops(plotdata.cov, result_min, plotdata.coords)


    dimops_mod, result_min_mod = modmissingdim(dimops, plotdata, result_min)

    # GET STRINGS
    title = get_title(input_f)
    file_str = get_file_string(title, dimops)

    if FIT:
        if result_min.status != 0:
            print("WARNING:  MINIMIZER FAILED TO CONVERGE AT LEAST ONCE")
        param_chisq = get_param_chisq(plotdata.coords, dimops, plotdata.fitcoord,
                                      result_min_mod, fitrange)
        print_messages(result_min_mod, param_err, param_chisq)

    # STOP IF NO PLOT
    if NO_PLOT:
        return 0

    # DO PLOT
    with PdfPages(file_str) as pdf:
        print("file name of saved plot:", file_str)
        plot_errorbar(dimops, xcoord, ycoord, error2)

        #plot dispersion analysis
        if PLOT_DISPERSIVE:
            plot_dispersive(dimops, xcoord)

        if FIT:
            # plot fit function
            plot_fit(plotdata.fitcoord, result_min_mod, dimops)

            # tolerance box plot
            if EFF_MASS and BOX_PLOT:
                plot_box(plotdata.fitcoord, result_min_mod, param_err,
                         dimops)

            annotate(dimops_mod, result_min_mod, param_err,
                     param_chisq, plotdata.coords)

        # save, output
        do_plot(title, pdf)

    return 0

def modmissingdim(dimops, plotdata, result_min):
    """Determine if we leave out the top state of the GEVP from fit
    then return the modified fit dimension and truncated result
    """
    xfit_check = get_xfit(dimops, plotdata.fitcoord) # determines if we left out a gevp dimension
    todel = []
    dimops_mod = dimops
    result_min_mod = result_min
    for i, j in enumerate(xfit_check):
        if j == []:
            dimops_mod -= 1
            todel.append(i)
    if FIT:
        # delete the unwanted dimensions
        for idx in todel:
            #result_min_mod.x = np.delete(result_min.x, todel)
            result_min_mod.x[idx] = np.nan
    return dimops_mod, result_min_mod


def plot_dispersive(dimops, xcoord):
    """Plot lines corresponding to dispersive analysis energies"""
    for i, energy in enumerate(DISP_ENERGIES):
        estring = trunc_prec(energy)
        """
        plt.annotate(
            "Dispersive energy["+str(i)+"] = "+estring,
            xy=(0.05, 0.90-(i+dimops)*.05), xycoords='axes fraction')
        """
        plt.plot(xcoord, list([energy])*len(xcoord),
                 label='Dispersive('+str(i)+')')
    plt.legend(loc='lower left')

def get_prelim_errbars(result_min):
    """If the fit range is not identical to the plot range,
    we are forced to use the traditional error bar method
    (although this case is already handled by this point in singlefit), i.e.
    the (jackknifed) average covariance matrix.
    Otherwise, defer to the user preference (from config).
    """
    try:
        error2 = np.array(result_min.error_bars)
    except AttributeError:
        error2 = None
    if ERROR_BAR_METHOD == 'avgcov':
        error2 = None #since other method not well understood
    elif ERROR_BAR_METHOD == 'jk':
        pass
    else:
        print("mkplot:Bad error bar method specified:", ERROR_BAR_METHOD)
        sys.exit(1)
    return error2


def get_dimops(cov, result_min, coords):
    """Get dimension of GEVP matrix or return 1 if not GEVP
    """
    try:
        dimops = len(cov[0][0])
    except TypeError:
        dimops = 1
    coords = np.array(coords)
    if FIT:
        dimops_chk = len(fit_func(
            coords[0][0], result_min.x)) if not isinstance(fit_func(
                coords[0][0], result_min.x), Number) else 1
        if dimops != dimops_chk:
            print("***ERROR***")
            print("Fit function length does not match cov. mat.")
            print("Debug of config necessary.")
            print(dimops, dimops_chk)
            sys.exit(1)
    return dimops


def get_title(input_f):
    """get title info"""
    # title/filename stuff
    if TITLE == '' or not TITLE:
        # then plot title should be the location directory of the jk blocks
        cwd = os.getcwd()
        if os.path.isdir(input_f):
            os.chdir(input_f)
            title = os.getcwd().split('/')[-1]
            os.chdir(cwd)
        else:
            title = input_f
    else:
        title = TITLE
    pretitle = latfit.config.TITLE_PREFIX+str(NUM_CONFIGS)+' configs '
    if len(pretitle) > 50:
        title = ''
        pretitle = pretitle[:-1]
    title = pretitle+title
    title = re.sub(r'_(?!{)', ' ', title) # don't get rid of latex subscripts
    if STYPE == 'hdf5':
        title = re.sub('.jkdat', '', title)
    return title


def get_file_string(title, dimops):
    """get strings"""

    # brief attempt at sanitization
    title_safe = re.sub('_', ' ', title)
    title_safe = re.sub(r'\(', '', title_safe)
    title_safe = re.sub(r'\)', '', title_safe)
    title_safe = re.sub(r'\$', '', title_safe)
    title_safe = re.sub(r'\\', '', title_safe)
    title_safe = re.sub(r', ', ' ', title_safe)
    title_safe = re.sub(r'vec{p} {CM}=', 'mom', title_safe)

    if JACKKNIFE_FIT == 'DOUBLE':
        jk_str = '_2xjk'
    elif JACKKNIFE_FIT == 'FROZEN':
        jk_str = '_1xjk'
    else:
        jk_str = ''
    if EFF_MASS:
        eff_str = '_eff_mass'
        if EFF_MASS_METHOD != 4:
            eff_str += '_meth'+str(EFF_MASS_METHOD)
        if EFF_MASS_METHOD == 1:
            print("C = ", C)
    else:
        eff_str = ''
    if UNCORR:
        print("Doing uncorrelated fit.")
        uncorr_str = '_uncorr_fit'
    else:
        uncorr_str = ''
    file_str = title_safe + eff_str + uncorr_str + jk_str
    file_str = re.sub(' ', '_', file_str) + '.pdf'
    return file_str


def get_coord(coords, cov, error2=None):
    """Plotted coordinates setup
    """
    print("list of plotted points [x, y]:")
    print(coords)
    xcoord = [coords[i][0] for i in range(len(coords))]
    ycoord = [coords[i][1] for i in range(len(coords))]
    if error2 is None:
        if GEVP:
            error2 = np.array([np.sqrt(np.diag(cov[i][i]))
                               for i in range(len(coords))])
        else:
            error2 = np.array([
                np.sqrt(cov[i][i]) for i in range(len(coords))])
    print("list of point errors (x, yerr):")
    for i, j in zip(xcoord, error2):
        print(i, j)
    return xcoord, ycoord, error2


def print_messages(result_min, param_err, param_chisq):
    """print message up here because of weirdness with pdfpages
    """
    startp = np.array(START_PARAMS)
    # print plot info
    print("Minimizer thinks that it worked.  Plotting fit.")
    print("Fit info:")
    print("Autofit:", AUTO_FIT)
    print("Model includes additive constant:", ADD_CONST)
    if EFF_MASS:
        print("Effective mass method:", EFF_MASS_METHOD)
    print("Assisted Fit:", ASSISTED_FIT)
    print("Minimizer (of chi^2) method:", METHOD)
    if METHOD == 'L-BFGS-B':
        print("Bounds:", BINDS)
    print("Guessed params:  ", np.array2string(startp, separator=', '))
    if EFF_MASS:
        print("Energies (MeV):", np.array2string(
            1000*AINVERSE*np.array(result_min.x), separator=', '))
        print("Error in energies (MeV):", np.array2string(
            1000*AINVERSE*np.array(param_err), separator=', '))
    else:
        print("Minimized params:", np.array2string(
            result_min.x, separator=', '))
        print("Error in params :", np.array2string(np.array(param_err),
                                                   separator=', '))
    chisq_str = str(result_min.fun)
    if JACKKNIFE_FIT:
        chisq_str += '+/-'+str(result_min.chisq_err)
    print("chi^2 minimized = ", chisq_str)
    print("degrees of freedom = ", param_chisq.dof)
    if (JACKKNIFE_FIT == 'DOUBLE' or JACKKNIFE_FIT == 'SINGLE') and \
       JACKKNIFE == 'YES':
        print("avg p-value = ", result_min.pvalue, "+/-",
              result_min.pvalue_err)
        print("p-value of avg chi^2 = ", 1 - stats.chi2.cdf(result_min.fun,
                                                            param_chisq.dof))
    redchisq_str = str(param_chisq.redchisq)
    print("chi^2/dof = ", redchisq_str)
    if CALC_PHASE_SHIFT:
        if GEVP:
            print("sqrt(s), I="+str(ISOSPIN)+" phase shift(in degrees) = ")
            energy = 1000*AINVERSE*np.array(result_min.x)
            mom = 1000*np.sqrt(AINVERSE**2*rf.norm2(rf.procmom(MOMSTR))*(2*np.pi/L_BOX)**2)
            print("mom =", mom)
            root_s = np.sqrt(energy**2-mom**2)
            err_energy = 1000*AINVERSE*np.array(param_err)*energy/root_s
            for i in range(len(result_min.scattering_length)):
                shift = result_min.phase_shift[i]
                shift = np.real(shift) if np.isreal(shift) else shift
                err = result_min.phase_shift_err[i]
                err = np.real(err) if np.isreal(err) else err
                print(root_s[i], "MeV :", "+/-", err_energy[i],
                      "phase shift:", shift, "+/-", err)
            print("[")
            for i in range(len(result_min.scattering_length)):
                print(np.array2string(np.array([root_s[i], shift, err_energy[i], err])), ",")
            print("]")
            for i in range(len(result_min.scattering_length)):
                if i == 0: # scattering length only meaningful as p->0
                    print("I="+str(ISOSPIN)+" scattering length = ",
                          result_min.scattering_length[i], "+/-",
                          result_min.scattering_length_err[i])
        else:
            print("I="+str(ISOSPIN)+" phase shift(in degrees) = ",
                  result_min.phase_shift, "+/-",
                  result_min.phase_shift_err)
            print("I="+str(ISOSPIN)+" scattering length = ",
                  result_min.scattering_length, "+/-",
                  result_min.scattering_length_err)


def get_param_chisq(coords, dimops, xcoord, result_min, fitrange=None):
    """Get chi^2 parameters."""
    param_chisq = namedtuple('param_chisq',
                             ('redchisq', 'redchisq_round_str', 'dof'))
    if fitrange is None:
        param_chisq.dof = int(len(coords)*dimops-len(result_min.x))
    else:
        param_chisq.dof = int((fitrange[1]-fitrange[0]+1)*dimops-len(
            result_min.x))
    # Do this because C parameter is a fit parameter,
    # it just happens to be guessed by hand
    if EFF_MASS and EFF_MASS_METHOD == 1 and C != 0.0:
        param_chisq.dof -= 1
    print("param_chisq.dof=", param_chisq.dof)
    print("FIT_EXCL=", latfit.config.FIT_EXCL)
    for k, i in enumerate(latfit.config.FIT_EXCL):
        if k >= len(result_min.x): # if we leave off a gevp dimension
            break
        for j in i:
            if j in xcoord:
                param_chisq.dof -= 1
    param_chisq.redchisq = result_min.fun/param_chisq.dof
    if JACKKNIFE_FIT:
        # redchisq_str = str(param_chisq.redchisq)
        # redchisq_str += '+/-'+str(result_min.chisq_err/param_chisq.dof)
        if (param_chisq.redchisq > 10 or param_chisq.redchisq < 0.1) or (
                result_min.chisq_err/param_chisq.dof > 10
                or result_min.chisq_err/param_chisq.dof < .1):
            param_chisq.redchisq_round_str = format_chisq_str(
                param_chisq.redchisq, plus=False)
        else:
            param_chisq.redchisq_round_str = format_chisq_str(
                param_chisq.redchisq, plus=True)
    return param_chisq


def format_chisq_str(chisq, plus=False):
    """Format the chi^2/dof string for plot annotation, jackknife fit"""
    formstr = '{:0.'+str(int(PREC_DISP))+'e}'
    form_str_plus = '{:0.'+str(int(PREC_DISP)+1)+'e}'
    if chisq >= 1 and chisq < 10:
        retstr = str(round(chisq, PREC_DISP))
    else:
        if plus:
            retstr = formstr.format(chisq)
        else:
            retstr = form_str_plus.format(chisq)
    return retstr

def plot_errorbar(dimops, xcoord, ycoord, error2):
    """plot data error bars
    """
    if dimops != 1:
        lcoord = len(xcoord)
        # for color-blind people,
        # make plot lines have (hopefully) unique markers
        marker = itertools.cycle(('o', 'X', 'd', 'p', 's'))
        for curve_num in range(dimops):
            ycurve = np.array([ycoord[i][curve_num]
                               for i in range(lcoord)])
            yerr = np.array([error2[i][curve_num] for i in range(lcoord)])
            plt.errorbar(xcoord, ycurve, yerr=yerr,
                         linestyle='None', ms=3.75, marker=next(marker),
                         label='Energy('+str(curve_num)+')')
    else:
        plt.errorbar(xcoord, ycoord, yerr=error2,
                     linestyle='None', ms=3.75, marker='o')


def plot_fit(xcoord, result_min, dimops):
    """Plot fit function
    the fit function is plotted on a scale FINE times more fine
    than the original data points (to show smoothness)
    """
    if EFF_MASS and EFF_MASS_METHOD == 3:
        pass
        # warn('step size assumed 1 for fitted plot.')
        # step_size = 1
    else:
        pass
    xfit = get_xfit(dimops, xcoord)
    for curve_num in range(dimops):
        # result_min.x is is the array of minimized fit params
        yfit = np.array([
            fit_func(xfit[curve_num][i], result_min.x)[curve_num] if dimops > 1 else
            fit_func(xfit[i], result_min.x)
            for i in range(len(xfit[curve_num] if dimops > 1 else xfit))])
        if np.nan in yfit:
            continue
        # only plot fit function if minimizer result makes sense
        # if result_min.status == 0:
        plt.plot(xfit[curve_num] if dimops > 1 else xfit, yfit)

def get_xfit(dimops, xcoord, step_size=None, box_plot=False):
    """Return the abscissa for the plot of the fit function."""
    if dimops == 1:
        step_size = abs((xcoord[len(xcoord)-1]-xcoord[0]))/FINE/(
            len(xcoord)-1)
        xfit = np.arange(xcoord[0], xcoord[len(xcoord)-1]+step_size,
                         step_size)
    else:
        xfit = np.zeros((dimops), dtype=object)
        for i in range(dimops):
            xfit[i] = np.array(xcoord)
            todel = []
            badcoord = []
            for j, coord in enumerate(xcoord):
                if coord in latfit.config.FIT_EXCL[i]:
                    todel.append(j)
                    badcoord.append(coord)
            xfit[i] = np.delete(xfit[i], todel)
            if not box_plot:
                step_size = abs((xfit[i][len(xfit[i])-1]-xfit[i][0]))/FINE/(
                    len(xfit[i])-1) if step_size is None else step_size
                step_size = 1.0 if np.isnan(step_size) else step_size
                try:
                    xfit[i] = list(np.arange(xfit[i][0],
                                            xfit[i][len(xfit[i])-1]+step_size,
                                            step_size))
                except IndexError: # here in case nothing is to be plot
                    xfit[i] = []
    return xfit


if GEVP:
    def plot_box(xcoord, result_min, param_err, dimops):
        """plot tolerance box around straight line fit for effective mass
        assumes xstep = 1
        """
        axvar = plt.gca()
        # gca, gcf = getcurrentaxes getcurrentfigure
        fig = plt.gcf()
        xfit = get_xfit(dimops, xcoord, 1, box_plot=True)
        xfit = [xfit] if dimops == 1 else xfit
        for i in range(dimops):
            if np.isnan(result_min.x[i]):
                continue
            try:
                continuous = len(
                    np.arange(xfit[i][0],
                              xfit[i][len(xfit[i])-1]+1)) == len(xfit[i])
                for start in xfit[i]:
                    delw = xfit[i][len(
                        xfit[i])-1]-xfit[i][0] if continuous else 0
                    axvar.add_patch((
                        plt.Rectangle(  # (11.0, 0.24514532441), 3,.001,
                            (start-.5, result_min.x[i]-param_err[i]),  # (x, y)
                            1+delw,  # width
                            2*param_err[i],  # height
                            fill=True, color='k', alpha=0.5,
                            zorder=1000, figure=fig,
                            # transform=fig.transFigure
                        )))
                    if continuous:
                        break
            except IndexError:
                pass
else:
    def plot_box(xcoord, result_min, param_err, dimops=1):
        """plot tolerance box around straight line fit for effective mass
        """
        if dimops:
            pass
        axvar = plt.gca()
        # gca, gcf = getcurrentaxes getcurrentfigure
        fig = plt.gcf()
        axvar.add_patch((
            plt.Rectangle(  # (11.0, 0.24514532441), 3,.001,
                (xcoord[0]-.5, result_min.x[0]-param_err[0]),   # (x, y)
                xcoord[len(xcoord)-1]-xcoord[0]+1,  # width
                2*param_err[0],          # height
                fill=True, color='k', alpha=0.5, zorder=1000, figure=fig,
                # transform=fig.transFigure
            )))

if ADD_CONST:
    YSTART = 0.95
else:
    YSTART = 0.35

if GEVP:
    if ADD_CONST:
        YSTART = 0.90
    else:
        YSTART = 0.90
    if EFF_MASS:
        YSTART = 0.2

    def annotate_energy(result_min, param_err, ystart=YSTART):
        """Annotate plot with fitted energy (GEVP)
        """
        # annotate plot with fitted energy
        nplots = len(param_err)+(len(DISP_ENERGIES) if PLOT_DISPERSIVE else 0)
        print('nplots=', nplots)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=nplots)
        #plt.legend(bbox_to_anchor=(1.24,1),loc='best')
        for i, min_e in enumerate(result_min.x):
            estring = trunc_prec(min_e)+"+/-"+trunc_prec(param_err[i], 2)
            plt.annotate(
                "Energy["+str(i)+"] = "+estring,
                xy=(0.05, ystart-i*.05), xycoords='axes fraction')
else:
    if ADD_CONST or not EFF_MASS:
        YSTART = 0.90
    else:
        YSTART = 0.35

    def annotate_energy(result_min, param_err, ystart=YSTART):
        """Annotate plot with fitted energy (non GEVP)
        """
        if len(result_min.x) > 1:
            estring = trunc_prec(result_min.x[1])+"+/-"+trunc_prec(
                param_err[1], 2)
        else:
            # for an effective mass plot
            estring = trunc_prec(result_min.x[0])+"+/-"+trunc_prec(
                param_err[0], 2)
        plt.annotate("Energy="+estring, xy=(0.05, ystart),
                     xycoords='axes fraction')


def trunc_prec(num, extra_trunc=0):
    """Truncate the amount of displayed digits of precision to PREC_DISP"""
    formstr = '%.'+str(int(PREC_DISP-extra_trunc))+'e'
    return str(float(formstr % Decimal(str(num))))


if EFF_MASS and EFF_MASS_METHOD == 3:
    if GEVP:
        if ADD_CONST:
            YSTART2 = 0.75
        else:
            YSTART2 = 0.75

        def annotate_chisq(redchisq_round_str, dof,
                           result_min, ystart=YSTART2):
            """Annotate with resultant chi^2 (eff mass, eff mass method 3)
            """
            rcp = r"$\chi^2$/dof = "
            rcp += redchisq_round_str+", dof = "+str(dof)
            plt.annotate(rcp, xy=(0.15, ystart-.05*(len(result_min.x)-2)),
                         xycoords='axes fraction')

    else:
        if ADD_CONST:
            YSTART2 = 0.10
        else:
            YSTART2 = 0.10

        def annotate_chisq(redchisq_round_str, dof,
                           result_min, ystart=YSTART2):
            """Annotate with resultant chi^2 (eff mass, eff mass method 3)
            """
            if result_min:
                pass
            rcp = r"$\chi^2$/dof = "
            rcp += redchisq_round_str+", dof = "+str(dof)
            plt.annotate(rcp, xy=(0.15, ystart),
                         xycoords='axes fraction')


else:

    def annotate_chisq(redchisq_round_str, dof, result_min=None):
        """Annotate with resultant chi^2
        """
        if result_min:
            pass
        plt.annotate(
            r"$\chi^2$/dof="+redchisq_round_str+", dof="+str(dof),
            xy=(0.5, 0.05),
            xycoords='axes fraction')


if JACKKNIFE_FIT:
    if JACKKNIFE_FIT == 'FROZEN':

        def annotate_jack():
            """Annotate jackknife type (frozen)"""
            plt.annotate('Frozen (single) jackknife fit.', xy=(
                0.05, 0.15), xycoords='axes fraction')

    elif JACKKNIFE_FIT == 'SINGLE':

        def annotate_jack():
            """Annotate jackknife type (single elim)"""
            plt.annotate('Single jackknife fit.', xy=(
                0.05, 0.15), xycoords='axes fraction')

    elif JACKKNIFE_FIT == 'DOUBLE':

        def annotate_jack():
            """Annotate jackknife type (double)"""
            pass
            #plt.annotate('Double jackknife fit.', xy=(
            #    0.10, 0.95), xycoords='axes fraction')


elif JACKKNIFE:

    def annotate_jack():
        """Annotate jackknife type (only avg)"""
        plt.annotate('Avg. fit, jackknife est. cov. matrix',
                     xy=(0.05, 0.15), xycoords='axes fraction')


else:

    def annotate_jack():
        """Annotate jackknife type (none)"""
        pass


if UNCORR:

    def annotate_uncorr(coords, dimops):
        """Annotate plot with uncorr"""
        ldisp = len(DISP_ENERGIES)
        if dimops > 1:
            plt.annotate("Uncorrelated fit.", xy=(0.05,
                                                  0.90-(ldisp+dimops)*0.05),
                         xycoords='axes fraction')
        else:
            plt.text(coords[3][0], coords[2][1], "Uncorrelated fit.")


else:

    def annotate_uncorr(*args):
        """Annotate plot with uncorr"""
        return args


def do_plot(title, pdf):
    """Do the plot, given the title."""
    # setup fonts
    hfontt = {'fontname': 'FreeSans', 'size': 12}
    hfontl = {'fontname': 'FreeSans', 'size': 14}
    # add axes labels, title
    plt.title(title, **hfontt)
    plt.xlabel(XLABEL, **hfontl)
    plt.ylabel(YLABEL, **hfontl)
    if EFF_MASS:
        ymin, ymax = plt.ylim()
        ymax = min(AINVERSE, ymax)
        ymin = max(0, ymin-0.2)
        plt.ylim(ymin, ymax)
    # read out into a pdf
    pdf.savefig()
    # show the plot
    plt.show()


def annotate(dimops, result_min, param_err, param_chisq, coords):
    """Annotate plot.
    param_chisq=[redchisq, redchisq_round_str, dof]
    """
    annotate_energy(result_min, param_err)
    # if result_min.status == 0 and param_chisq.redchisq < 2:
    if param_chisq.redchisq < 2:
        annotate_chisq(param_chisq.redchisq_round_str,
                       param_chisq.dof, result_min)
    annotate_jack()
    annotate_uncorr(coords, dimops)
