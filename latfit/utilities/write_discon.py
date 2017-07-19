#!/usr/bin/python3
"""Write disconnected diagrams"""
import os.path
from os import listdir
from os.path import isfile, join
import re
import numpy as np

import read_file as rf
import combine as cb

#gets the array from the file, but keeps the values as strings
def comb_fig(dsrc, dsnk):
    """Get combined figure name from bubble figure names."""
    figsrc = rf.figure(dsrc)
    figsnk = rf.figure(dsnk)
    if figsrc == 'scalar-bubble' and figsnk == 'scalar-bubble':
        return 'Bub2'
    elif figsrc == 'scalar-bubble' and figsnk == 'Vdis':
        return 'Cv3'
    elif figsrc == 'Vdis' and figsnk == 'scalar-bubble':
        return 'Cv3R'
    elif figsrc == 'Vdis' and figsnk == 'Vdis':
        return 'V'

def single_p(ptest):
    """is the momentum array only a single momentum?"""
    return bool((len(ptest) == 3 and isinstance(ptest[0], int)))

def momtotal(mom):
    """Find total center of mass momenta from momenta array"""
    if single_p(mom):
        momret = mom
    else:
        mom1 = np.array(mom[0])
        mom2 = np.array(mom[1])
        momret = mom1+mom2
    return momret

def dismom(psrc, psnk):
    """Get combined momentum string from disconnected momenta"""
    lenp = len(psrc)+len(psnk)
    if lenp == 4:
        #V
        mom1src = psrc[0]
        mom2src = psrc[1]
        #reverse meaning of inner and outer, so take [1] for inner
        mom1snk = psnk[1]
        momstr = "mom1src"+rf.ptostr(
            mom1src)+"_mom2src"+rf.ptostr(
                mom2src)+"_mom1snk"+rf.ptostr(mom1snk)
    elif lenp == 5:
        #Cv3
        if single_p(psrc):
            momsrc = psrc
            #reverse meaning of inner and outer, so take [1] for inner
            momsnk = psnk[1]
        elif single_p(psnk):
            momsnk = psnk
            momsrc = psrc[0]
        momstr = "momsrc"+rf.ptostr(momsrc)+"_momsnk"+rf.ptostr(momsnk)
    elif lenp == 6:
        #Bub2
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
        if not fign in ["scalar-bubble", "Vdis"]:
            continue
        traj = rf.traj(filen)
        mom = rf.mom(filen)
        file_lookup.setdefault(traj, {}).setdefault(
            rf.ptostr(momtotal(mom)), []).append((filen, mom))
    return file_lookup

def main():
    """Write disconnected diagrams, main"""
    dur = 'summed_tsrc_diagrams/'
    lookup = {}
    file_lookup = get_disfiles([
        f for f in listdir('.') if isfile(join('.', f))])
    for traj in file_lookup:
        for mt1 in file_lookup[traj]:
            for dsrc, momsrc in file_lookup[traj][mt1]:
                for dsnk, momsnk in file_lookup[traj][mt1]:
                    outfig = comb_fig(dsrc, dsnk)
                    if not outfig:
                        continue
                    try:
                        sepstr, sepval = get_sep(dsrc, dsnk, outfig)
                    except TypeError:
                        continue
                    outstr = "_Figure"+outfig+sepstr+dismom(momsrc, momsnk)
                    outfile = "traj_"+str(
                        traj)+outstr
                    outavg = "AvgVac"+outstr
                    if os.path.isfile(outfile) and os.path.isfile(outavg):
                        print("Skipping:", outfile, outavg)
                        print("File exists.")
                        continue
                    arr_plus, arr_minus = get_data(dsrc, dsnk, sepval,
                                                   dur, lookup)
                    #rf.write_arr(arr_plus - arr_minus, outfile)
                    rf.write_arr(arr_plus, outfile)
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
            #sepval = sep
            #we do this because src pipi bubbles don't need a
            #separation offset when combining
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

def get_data(dsrc, dsnk, sepval, dur, lookup):
    """Get regular data and vac subtraction diagram"""
    #get the data
    #Note:  cb.comb_dis defaults to taking the complex conjugate of src only.
    arr_plus = np.array(cb.comb_dis(dsrc, dsnk, sepval))
    src_fig = rf.figure(dsrc)
    snk_fig = rf.figure(dsnk)
    dsrc_sub = re.sub(src_fig, "Avg_"+src_fig, dur+dsrc)
    dsnk_sub = re.sub(snk_fig, "Avg_"+snk_fig, dur+dsnk)
    dsrc_sub = re.sub(r'traj_(\d)+_Figure_', '', dsrc_sub)
    dsnk_sub = re.sub(r'traj_(\d)+_Figure_', '', dsnk_sub)
    #get the  <><> subtraction array (<> indicates avg over trajectories)
    if dsrc_sub+dsnk_sub in lookup:
        print("Using prev.")
        arr_minus = lookup[dsrc_sub+dsnk_sub]
    else:
        arr_minus = np.array(cb.comb_dis(dsrc_sub, dsnk_sub, sepval))
        lookup[dsrc_sub+dsnk_sub] = arr_minus
    return arr_plus, arr_minus

if __name__ == "__main__":
    main()
