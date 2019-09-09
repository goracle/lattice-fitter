"""Calculate phase shift from Luscher zeta for given inputs."""
import sys
import subprocess
import inspect
import os
import math
from math import sqrt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from accupy import kdot
from latfit.config import PION_MASS, L_BOX, CALC_PHASE_SHIFT
from latfit.config import AINVERSE, ISOSPIN, MOMSTR, FIT_SPACING_CORRECTION
from latfit.config import IRREP
from latfit.utilities import read_file as rf
from latfit.analysis.errorcodes import ZetaError, RelGammaError
import latfit.utilities.zeta.i1zeta as i1z

if CALC_PHASE_SHIFT:
    def remove_epipi_indexing(epipi):
        """Remove the indexining on epipi"""
        try:
            epipi = epipi[1]
        except (IndexError, TypeError):
            try:
                epipi = epipi[0]
            except (IndexError, TypeError):
                pass
        return epipi

    def zeta(epipi):
        """Calculate the I=0 scattering phase shift given the pipi energy
        for that channel.
        """
        epipi = remove_epipi_indexing(epipi)
        comp = np.array(rf.procmom(MOMSTR))
        gamma = 1
        if epipi:
            try:
                if FIT_SPACING_CORRECTION:
                    arg = epipi**2-(2*np.pi/L_BOX)**2*kdot(comp, comp)
                    gamma = epipi/sqrt(arg)
                else:
                    arg = epipi**2-4*np.sin(
                        np.pi/L_BOX)**2*kdot(comp, comp)
                    gamma = epipi/sqrt(arg)
            except (ValueError, FloatingPointError):
                print("zeta.py, bad gamma value for epipi=", epipi)
                print("arg=", arg)
                print("center of mass momentum=", comp)
                print("Length of box=", L_BOX)
                raise ZetaError("bad gamma, epipi = "+str(epipi))
        if gamma < 1:
            raise RelGammaError(gamma=gamma, epipi=epipi)
        epipi = epipi*AINVERSE/gamma
        lbox = L_BOX/AINVERSE
        #epipi = math.sqrt(epipi**2-(2*np.pi/lbox)**2*PTOTSQ) //not correct

        # set up the normal call to w00 phase shift method
        binpath = os.path.dirname(inspect.getfile(zeta))+'/main.o'
        arglist = [binpath, str(epipi), str(PION_MASS), str(lbox),
                   str(comp[0]), str(comp[1]), str(comp[2]), str(gamma),
                   str(int(not FIT_SPACING_CORRECTION))]

        # set up the I=1 moving frame version
        i1z.COMP = MOMSTR
        i1z.L_BOX = np.float(lbox)
        i1z.IRREP = str(IRREP)
        i1z.MPION = np.float(PION_MASS)

        try:
            if not np.isnan(epipi):
                if ISOSPIN != 1 or not np.any(comp):
                    out = subprocess.check_output(arglist)
                else:
                    out = i1z.phase(epipi)
            else:
                out = np.nan
        except FileNotFoundError:
            print("Error in zeta: main.C not compiled yet.")
            print(subprocess.check_output(['pwd']))
            print(inspect.getfile(zeta))
            sys.exit(1)
        except subprocess.CalledProcessError:
            print("Error in zeta: calc of phase shift error:")
            print(epipi)
            errstr = subprocess.Popen(arglist,
                                      stdout=subprocess.PIPE).stdout.read()
            raise ZetaError(errstr)
        try:
            test = epipi*epipi/4-PION_MASS**2 < 0
        except FloatingPointError:
            print("floating point error")
            print("epipi=", epipi)
            sys.exit(1)
        if test:
            out = float(out)*1j
        else:
            try:
                out = complex(float(out))
            except ValueError:
                print("unable to convert phase shift to number:", out)
                print("check to make sure there does not exist"+\
                      " debugging which needs to be turned off.")
                print(out)
                raise ZetaError("bad number conversion")
            if ISOSPIN == 0:
                if out.real < 0 and abs(out.real) > 90:
                    out = np.complex(out.real+
                                     math.ceil(-1*out.real/180)*180, out.imag)
                if out.real > 180:
                    out = np.complex(out.real-
                                     math.floor(out.real/180)*180, out.imag)
        return out
else:
    def zeta(_):
        """Blank function; do not calculate phase shift"""
        return

if CALC_PHASE_SHIFT:
    def pheno(epipi):
        """Calc pheno phase shift"""
        try:
            epipi = epipi[1]
        except (IndexError, TypeError):
            try:
                epipi = epipi[0]
            except (IndexError, TypeError):
                pass
        epipi = epipi*AINVERSE
        binpath = os.path.dirname(inspect.getfile(zeta))+'/pheno.o'
        arglist = [binpath, str(epipi), str(PION_MASS), str(ISOSPIN)]
        try:
            out = subprocess.check_output(arglist)
        except FileNotFoundError:
            print("Error in pheno: pheno.C not compiled yet.")
            print(subprocess.check_output(['pwd']))
            print(inspect.getfile(zeta))
            sys.exit(1)
        except subprocess.CalledProcessError:
            print("Error in pheno: calc of phase shift error:")
            print(epipi)
            errstr = subprocess.Popen(arglist,
                                      stdout=subprocess.PIPE).stdout.read()
            raise ZetaError(errstr)
        return float(out)


else:
    def pheno(_):
        """Blank function; do not calculate phase shift"""
        return

def zeta_real(epipi):
    """Gives nan's if zeta(E_pipi) is not real"""
    test = zeta(epipi)
    if test.imag != 0:
        retval = math.nan
    else:
        retval = test.real
    return retval


def plotcrosscurves(plot_both=False):
    """Plots the cross curves of Luscher and Schenk (pheno)
    the intersection points are predictions for lattice energies
    """
    points = 1e3 # Number of points
    xmin, xmax = 2*float(PION_MASS)/AINVERSE, 1.1
    xlist = list(map(lambda x: float(xmax - xmin)*1.0*x/(
        points*1.0)+xmin, list(np.arange(points+1))))
    xlist = [0+0j if np.isnan(i) else i for i in xlist]
    #ylist_pheno_minus = list(map(lambda y: -pheno(y), xlist))
    #plt.plot(xlist, ylist_pheno_minus, label='pheno-')
    #plt.plot(xlist, ylist_pheno_plus, label='pheno+')
    hfontt = {'fontname': 'FreeSans', 'size': 12}
    hfontl = {'fontname': 'FreeSans', 'size': 14}
    print('Isospin=', ISOSPIN)
    with PdfPages('SchenkVLuscherI'+str(ISOSPIN)+'.pdf') as pdf:
        if plot_both:
            ylist_pheno_plus = list(map(pheno, xlist))
            ylist_zeta = list(map(zeta, xlist))
            plt.plot(xlist, ylist_pheno_plus, label='Schenk')
            plt.plot(xlist, ylist_zeta, label='Luscher')
        else:
            ylist_dif = list(map(lambda y: zeta_real(y)-pheno(y), xlist))

            plt.plot(xlist, np.zeros((len(xlist))))
            plt.plot(xlist, ylist_dif, label='Luscher-Schenk')
        plt.title('Luscher Method, Schenk Phenomenology, Isospin='+
                  str(ISOSPIN), **hfontt)
        plt.xlabel('Ea (Lattice Units, a^(-1)='+
                   str(AINVERSE)+' GeV)', **hfontl)
        plt.ylabel(r'$\delta$ (degrees)', **hfontl)
        plt.legend(loc='best')
        pdf.savefig()
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),  shadow=True, ncol=nplots)
        plt.show()
