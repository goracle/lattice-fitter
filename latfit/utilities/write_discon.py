#!/usr/bin/python3
"""Write disconnected diagrams"""
import sys
import os.path
from os import listdir
from os.path import isfile, join
from collections import namedtuple
import re
import numpy as np
from latfit.utilities import exactmean as em

import latfit.utilities.read_file as rf
import latfit.utilities.combine as cb

# gets the array from the file, but keeps the values as strings


def comb_fig(dsrc, dsnk):
    """Get combined figure name from bubble figure names."""
    figsrc = rf.figure(dsrc) if 'type' not in  dsrc and 'mix' not in dsrc else dsrc
    figsnk = rf.figure(dsnk)
    if figsrc == 'scalar-bubble' and figsnk == 'scalar-bubble':
        retval = 'Bub2'
    elif figsrc == 'scalar-bubble' and figsnk == 'Vdis':
        retval = 'Cv3R'
    elif figsrc == 'Vdis' and figsnk == 'scalar-bubble':
        retval = 'Cv3'
    elif figsrc == 'Vdis' and figsnk == 'Vdis':
        retval = 'V'
    elif figsrc == 'kk-bubble' and figsnk == 'Vdis':
        retval = 'VKK2pipi'
    elif figsrc == 'kk-bubble' and figsnk == 'kk-bubble':
        retval = 'VKK2KK'
    elif figsrc == 'kk-bubble' and figsnk == 'scalar-bubble':
        retval = 'VKK2sigma'
    elif 'type4' in figsrc and figsnk == 'Vdis':
        retval = 'type4'
    elif 'type4' in figsrc and figsnk == 'scalar-bubble':
        retval = 'type4sigma'
    else:
        print("***ERROR***")
        print("write_discon:comb_fig: naming error:", figsrc, figsnk)
        print("from inputs:", dsrc, dsnk)
        sys.exit(1)
    return retval


def single_p(ptest):
    """is the momentum array only a single momentum?"""
    ret = bool((len(ptest) == 3 and isinstance(ptest[0], (int, np.integer))))
    if ret:
        try:
            int(ptest[2])
        except AssertionError:
            print("bad momentum:", ptest)
            raise
    return ret


def momtotal(mom):
    """Find total center of mass momenta from momenta array"""
    if single_p(mom):
        momret = mom
    else:
        mom1 = np.array(mom[0])
        try:
            mom2 = np.array(mom[1])
        except IndexError:
            print("bad momentum spec:", mom)
            raise
        momret = mom1+mom2
    return momret


def dismom(psrc, psnk):
    """Get combined momentum string from disconnected momenta"""
    lenp = len(psrc)+len(psnk)
    if lenp == 4:
        # V
        mom1src = psrc[0]
        mom2src = psrc[1]
        # reverse meaning of inner and outer, so take [1] for inner
        mom1snk = -1*np.array(psnk[1])  # complex conj at sink
        momstr = "mom1src"+rf.ptostr(
            mom1src)+"_mom2src"+rf.ptostr(
                mom2src)+"_mom1snk"+rf.ptostr(mom1snk)
    elif lenp == 5:
        # Cv3
        if single_p(psrc):
            momsrc = psrc
            # reverse meaning of inner and outer, so take [1] for inner
            momsnk = -1*np.array(psnk[1])  # complex conjugate at sink
        elif single_p(psnk):
            momsnk = -1*np.array(psnk)  # complex conjugate at sink
            momsrc = psrc[0]
        momstr = "momsrc"+rf.ptostr(momsrc)+"_momsnk"+rf.ptostr(momsnk)
    elif lenp == 6:
        # Bub2
        momstr = "mom"+rf.ptostr(psrc)
    else:
        print("Error: bad momenta:", psrc, psnk)
        sys.exit(1)
    return momstr


def get_disfiles(onlyfiles):
    """Get bubbles."""
    file_lookup = {}
    for filen in onlyfiles:
        fign = rf.figure(filen)
        if fign not in ["scalar-bubble", "Vdis"]:
            continue
        traj = rf.traj(filen)
        mom = rf.mom(filen)
        file_lookup.setdefault(traj, {}).setdefault(
            rf.ptostr(momtotal(mom)), []).append((filen, mom))
    return file_lookup


ZERO = '000'


def main():
    """Write disconnected diagrams, main"""
    dur = 'summed_tsrc_diagrams/'
    lookup = {}
    sepdata = namedtuple('sep', ['sepstr', 'sepval'])
    file_lookup = get_disfiles([
        f for f in listdir('.') if isfile(join('.', f))])
    for traj in file_lookup:
        # count = 0
        for mt1 in file_lookup[traj]:
            lookup_local = {}
            for dsrc, momsrc in file_lookup[traj][mt1]:
                for dsnk, momsnk in file_lookup[traj][mt1]:
                    outfig = comb_fig(dsrc, dsnk)
                    try:
                        sepdata.sepstr, sepdata.sepval = get_sep(
                            dsrc, dsnk, outfig)
                    except TypeError:
                        continue
                    # count += 1
                    # print(count)
                    outfile = "traj_" + str(traj) + "_Figure" + outfig + \
                        sepdata.sepstr+dismom(
                            momsrc, momsnk)
                    outavg = outfile+'_avgsub'
                    if os.path.isfile(outavg):
                        print("Skipping:", outavg)
                        print("File exists.")
                        continue
                    # arr_plus, arr_minus = get_data(
                    arr_minus, lookup_local = get_data(
                        get_fourfn(dsrc, dsnk, dur), sepdata.sepval,
                        lookup, onlyreal=bool(mt1 == ZERO),
                        lookup_local=lookup_local)
                    # rf.write_arr(arr_plus - arr_minus, outfile)
                    # rf.write_arr(arr_plus, outfile)
                    rf.write_arr(arr_minus, outavg)


def get_sep(dsrc, dsnk, outfig):
    """Get time sep info"""
    sepsrc = rf.sep(dsrc)
    sepsnk = rf.sep(dsnk)
    if outfig == 'V' and sepsrc != sepsnk:
        retsep = None
    else:
        sepval = 0
        sep = None
        sepstr = "_"
        if sepsrc and not sepsnk:
            sep = sepsrc
            # sepval = sep
            # we do this because src pipi bubbles don't need a
            # separation offset when combining
            sepstr += "sep"+str(sep)+"_"
        elif sepsnk and not sepsrc:
            sep = sepsnk
            sepval = int(sep)
            sepstr += "sep"+str(sep)+"_"
        elif sepsnk and sepsrc:
            sep = sepsnk
            sepval = int(sep)
            sepstr += "sep"+str(sep)+"_"
        retsep = sepstr, sepval
    return retsep


def get_fourfn(dsrc, dsnk, dur):
    """Get average bubble names"""
    bubs = namedtuple('bubs', ['dsrc', 'dsnk', 'dsrc_sub', 'dsnk_sub'])
    bubs.dsrc = dsrc
    bubs.dsnk = dsnk
    src_fig = rf.figure(dsrc)
    snk_fig = rf.figure(dsnk)
    bubs.dsrc_sub = re.sub(src_fig, "Avg_"+src_fig, dur+dsrc)
    bubs.dsnk_sub = re.sub(snk_fig, "Avg_"+snk_fig, dur+dsnk)
    bubs.dsrc_sub = re.sub(r'traj_(\d)+_Figure_', '', bubs.dsrc_sub)
    bubs.dsnk_sub = re.sub(r'traj_(\d)+_Figure_', '', bubs.dsnk_sub)
    return bubs


def get_data(bubs, sepval, lookup, onlyreal=False, lookup_local=()):
    """Get regular data and vac subtraction diagram"""
    # get the data
    # Note:  cb.comb_dis defaults to taking the complex conjugate of src only.

    if bubs.dsrc_sub in lookup:
        arr_minus_src = lookup[bubs.dsrc_sub]
    else:
        arr_minus_src = rf.proc_vac_real(
            bubs.dsrc_sub) if onlyreal else rf.proc_vac(bubs.dsrc_sub)
        lookup[bubs.dsrc_sub] = arr_minus_src
    if bubs.dsnk_sub in lookup:
        arr_minus_snk = lookup[bubs.dsnk_sub]
    else:
        arr_minus_snk = rf.proc_vac_real(
            bubs.dsnk_sub) if onlyreal else rf.proc_vac(bubs.dsnk_sub)
        lookup[bubs.dsnk_sub] = arr_minus_snk

    if bubs.dsrc in lookup_local:
        src = lookup_local[bubs.dsrc]
    else:
        if onlyreal:
            src = rf.proc_vac_real(bubs.dsrc)
        else:
            src = rf.proc_vac(bubs.dsrc)
        lookup_local[bubs.dsrc] = src
    if bubs.dsnk in lookup_local:
        snk = lookup_local[bubs.dsnk]
    else:
        if onlyreal:
            snk = rf.proc_vac_real(bubs.dsnk)
        else:
            snk = rf.proc_vac(bubs.dsnk)
        lookup_local[bubs.dsnk] = snk

    print("combining:", bubs.dsrc, bubs.dsnk)
    print("sub bubs:", bubs.dsrc_sub, bubs.dsnk_sub)
    # arr_plus = cb.comb_dis(src, snk, sepval)
    arr_minus = cb.comb_dis(src-em.acmean(arr_minus_src),
                            snk-em.acmean(arr_minus_snk), sepval)
    # get the  <><> subtraction array (<> indicates avg over trajectories)
    # return arr_plus, arr_minus
    return arr_minus, lookup_local


if __name__ == "__main__":
    main()
