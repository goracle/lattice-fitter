"""Calculate phase shift from Luscher zeta for given inputs."""
import sys
import subprocess
import inspect
import os
import math
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from latfit.config import PION_MASS, L_BOX, CALC_PHASE_SHIFT, START_PARAMS, PTOTSQ, AINVERSE, ISOSPIN
import matplotlib.pyplot as plt

class ZetaError(Exception):
    def __init__(self, mismatch):
        Exception.__init__(self, mismatch)

if CALC_PHASE_SHIFT:
    def zeta(epipi):
        """Calculate the I=0 scattering phase shift given the pipi energy
        for that channel.
        """
        try:
            epipi = epipi[1]
        except (IndexError, TypeError):
            try:
                epipi = epipi[0]
            except (IndexError, TypeError):
                pass
        epipi = epipi*AINVERSE
        #epipi = math.sqrt(epipi**2-(2*np.pi/L_BOX)**2*PTOTSQ) //not correct
        binpath = os.path.dirname(inspect.getfile(zeta))+'/main.o'
        arglist = [binpath, str(epipi), str(PION_MASS), str(L_BOX)]
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
        if(epipi*epipi/4-PION_MASS**2<0):
            out = float(out)*1j
        else:
            out = complex(float(out))
            if ISOSPIN == 0:
                if out.real < 0:
                    out = np.complex(out.real+
                                    math.ceil(-out.real/180)*180, out.imag)
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
    test = zeta(epipi)
    if test.imag != 0:
        retval = math.nan
    else:
        retval = test.real
    return retval


def plotcrosscurves(plot_both=False):
    points = 1e3 # Number of points
    xmin, xmax= 0, 1.1
    xlist = list(map(lambda x: float(xmax - xmin)*1.0*x/(points*1.0), list(np.arange(points+1))))
    #ylist_pheno_minus = list(map(lambda y: -pheno(y), xlist))
    #plt.plot(xlist, ylist_pheno_minus, label='pheno-')
    #plt.plot(xlist, ylist_pheno_plus, label='pheno+')
    hfontt = {'fontname': 'FreeSans', 'size': 12}
    hfontl = {'fontname': 'FreeSans', 'size': 14}
    print('Isospin=', ISOSPIN)
    with PdfPages('SchenkVLuscherI'+str(ISOSPIN)+'.pdf') as pdf:
        if plot_both:
            ylist_pheno_plus = list(map(lambda y: pheno(y), xlist))
            ylist_zeta = list(map(lambda y: zeta(y), xlist))
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
        plt.ylabel('$\delta$ (degrees)', **hfontl)
        plt.legend(loc='best')
        pdf.savefig()
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),  shadow=True, ncol=nplots)
        plt.show()
