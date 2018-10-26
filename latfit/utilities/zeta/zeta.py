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
from latfit.config import PION_MASS, L_BOX, CALC_PHASE_SHIFT
from latfit.config import AINVERSE, ISOSPIN, MOMSTR, FIT_SPACING_CORRECTION
from latfit.utilities import read_file as rf

class ZetaError(Exception):
    """Define an error for generic phase shift calc failure"""
    def __init__(self, mismatch):
        Exception.__init__(self, mismatch)

class RelGammaError(Exception):
    """Exception for imaginary GEVP eigenvalue"""
    def __init__(self, gamma=None, epipi=None, message=''):
        print("***ERROR***")
        print("gamma < 1: gamma=", gamma, "Epipi=", epipi)
        super(RelGammaError, self).__init__(message)
        self.gamma = gamma
        self.epipi = epipi
        self.message = message

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
        try:
            if FIT_SPACING_CORRECTION:
                gamma = epipi/sqrt(
                    epipi**2-(2*np.pi/L_BOX)**2*np.dot(comp, comp))
            else:
                gamma = epipi/sqrt(
                    epipi**2-4*np.sin(np.pi/L_BOX)**2*np.dot(comp, comp))
        except ValueError:
            print("zeta.py, bad gamma value for epipi=", epipi)
            print("center of mass momentum=", comp)
            print("Length of box=", L_BOX)
            raise ZetaError("bad gamma, epipi = "+str(epipi))
        if gamma < 1:
            raise RelGammaError(gamma=gamma, epipi=epipi)
        epipi = epipi*AINVERSE/gamma
        lbox = L_BOX/AINVERSE
        #epipi = math.sqrt(epipi**2-(2*np.pi/lbox)**2*PTOTSQ) //not correct
        binpath = os.path.dirname(inspect.getfile(zeta))+'/main.o'
        arglist = [binpath, str(epipi), str(PION_MASS), str(lbox),
                   str(comp[0]), str(comp[1]), str(comp[2]), str(gamma),
                   str(int(not FIT_SPACING_CORRECTION))]
        try:
            out = subprocess.check_output(arglist)
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
        test = 2*np.arcsin(np.sqrt(epipi*epipi/4-PION_MASS**2)/2) < 0
        test2 = np.sqrt(epipi*epipi/4-PION_MASS**2) < 0
        test = test2 if FIT_SPACING_CORRECTION else test
        if test:
            out = float(out)*1j
        else:
            try:
                out = complex(float(out))
            except ValueError:
                print("unable to convert phase shift to number")
                print("check to make sure there does not exist"+\
                      " debugging which needs to be turned off.")
                sys.exit(1)
                # raise ZetaError("bad number conversion")
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
    xmin, xmax = 0, 1.1
    xlist = list(map(lambda x: float(xmax - xmin)*1.0*x/(points*1.0), list(np.arange(points+1))))
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
