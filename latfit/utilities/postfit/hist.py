#!/usr/bin/python3
"""Make histograms from fit results over fit ranges"""
import sys
import subprocess
import pickle
import numpy as np
import gvar
from latfit.config import ISOSPIN, LATTICE_ENSEMBLE
from latfit.utilities.postprod.h5jack import ENSEMBLE_DICT
from latfit.utilities.postfit.bin_analysis import process_res_to_best
from latfit.utilities.postfit.bin_analysis import print_tot
from latfit.utilities.postfit.bin_analysis import select_ph_en
from latfit.utilities.postfit.bin_analysis import consis_tot
from latfit.utilities.postfit.gather_data import enph_filenames
from latfit.utilities.postfit.gather_data import make_hist
from latfit.utilities.postfit.fitwin import set_tadd_tsub, LENMIN
from latfit.utilities.postfit.compare_print import print_sep_errors
from latfit.utilities.postfit.compare_print import print_compiled_res

TDIS_MAX = ENSEMBLE_DICT[LATTICE_ENSEMBLE]['tdis_max']

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

SYS_ALLOWANCE = None
#SYS_ALLOWANCE = [['0.44042(28)', '-3.04(21)'], ['0.70945(32)',
# '-14.57(28)'], ['0.8857(39)', '-19.7(4.7)']]

@PROFILE
def geterr(allow):
    """Process SYS_ALLOWANCE"""
    ret = allow
    if allow is not None:
        ret = []
        for _, item in enumerate(allow):
            print(item)
            ret.append([gvar.gvar(item[j]).sdev for j,
                        _ in enumerate(item)])
        ret = np.array(ret)
    return ret

SYS_ALLOWANCE = geterr(SYS_ALLOWANCE)




@PROFILE
def check_ids_hist():
    """Check the ensemble id file to be sure
    not to run processing parameters from a different ensemble"""
    ids_check = [ISOSPIN, LENMIN]
    ids_check = np.asarray(ids_check)
    ids = pickle.load(open('ids_hist.p', "rb"))
    assert np.all(ids == ids_check),\
        "wrong ensemble. [ISOSPIN, LENMIN]"+\
        " should be:"+str(ids)+" instead of:"+str(ids_check)
    return ids

@PROFILE
def main(nosave=True):
    """Make the histograms."""
    if len(sys.argv[1:]) == 1 and (
            'phase_shift' in sys.argv[1] or\
            'energy' in sys.argv[1]) and nosave:

        # init variables
        fname = sys.argv[1]
        tot = []
        tot_pr = []
        success_tadd_tsub = []
        tsub = 0
        breakadd = False

        # get file names
        energyfn, phasefn = enph_filenames(fname)

        # loop over fit windows
        while np.abs(tsub) < TDIS_MAX:
            tadd = 0
            while tadd < TDIS_MAX:
                select_ph_en('energy')
                set_tadd_tsub(tadd, tsub)
                min_en = make_hist(
                    energyfn, nosave=nosave, allowidx=0)
                if min_en:
                    select_ph_en('phase')
                    min_ph = make_hist(
                        phasefn, nosave=nosave,
                        allowidx=1)
                if min_en and min_ph: # check this
                    process_res_to_best(min_en, min_ph)
                    toapp, test, toapp_pr = print_compiled_res(
                        min_en, min_ph)
                    if test:
                        success_tadd_tsub.append((tadd, tsub))
                    else:
                        breakadd = not tadd
                        break
                    tot.append(toapp)
                    consis_tot(tot)
                    tot_pr.append(toapp_pr)
                else:
                    breakadd = not tadd
                    if breakadd:
                        if not min_en:
                            print("missing energies")
                        if not min_ph:
                            print("missing phases")
                    break
                tadd += 1
            if breakadd:
                print("stopping loop on tadd, tsub:",
                      tadd, tsub)
                break
            tsub -= 1
        print("Successful (tadd, tsub):")
        for i in success_tadd_tsub:
            print(i)
        subprocess.check_output(['notify-send', '-u',
                                 'critical', '-t', '30',
                                 'hist: tloop complete'])
        print_sep_errors(tot_pr)
        print_tot(tot)
    else:
        for fname in sys.argv[1:]:
            min_res = make_hist(fname, nosave=nosave)
        print("minimized error results:", min_res)

if __name__ == '__main__':
    try:
        open('ids_hist.p', 'rb')
    except FileNotFoundError:
        print("be sure to set ISOSPIN, LENMIN",
              "correctly")
        print("edit histogram.py to remove the hold then rerun")
        # the hold
        # sys.exit(1)
        IDS_HIST = [ISOSPIN, LENMIN]
        IDS_HIST = np.asarray(IDS_HIST)
        pickle.dump(IDS_HIST, open('ids_hist.p', "wb"))
    check_ids_hist()
    main()
