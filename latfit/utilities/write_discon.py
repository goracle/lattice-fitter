#!/usr/bin/python3
"""Write disconnected diagrams"""
import os.path
from os import listdir
from os.path import isfile, join
import re
import numpy as np
import read_file as rf
from traj_list import traj_list
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
        exit(1)
    return momstr

def get_disfiles(traj, onlyfiles):
    """Get bubbles for a given trajectory."""
    disfiles = []
    for filen in onlyfiles:
        if rf.traj(filen) != traj:
            continue
        fign = rf.figure(filen)
        if not fign in ["scalar-bubble", "Vdis"]:
            continue
        disfiles.append(filen)
    return disfiles

def main():
    """Write disconnected diagrams, main"""
    dur = 'summed_tsrc_diagrams/'
    onlyfiles = [f for f in listdir('.') if isfile(join('.', f))]
    lookup = {}
    for traj in traj_list(onlyfiles):
        disfiles = get_disfiles(traj, onlyfiles)
        for dsrc in disfiles:
            for dsnk in disfiles:
                momsrc = rf.mom(dsrc)
                momsnk = rf.mom(dsnk)
                if not np.array_equal(momtotal(momsrc), momtotal(momsnk)):
                    continue
                outfig = comb_fig(dsrc, dsnk)
                if not outfig:
                    continue
                try:
                    sepstr, sepval = get_sep(dsrc, dsnk, outfig)
                except TypeError:
                    continue
                outfile = "traj_"+str(traj)+"_Figure"+outfig+sepstr+dismom(
                    momsrc, momsnk)
                if os.path.isfile(outfile):
                    print("Skipping:", outfile)
                    print("File exists.")
                    continue
                arr_plus, arr_minus = get_data(dsrc, dsnk,
                                               sepval, dur, lookup)
                rf.write_arr(arr_plus - arr_minus, outfile)
                #rf.write_arr(arr_plus, outfile)

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
        if sepsrc:
            sep = sepsrc
            #sepval = 0
            #we do this because src pipi bubbles don't need a
            #separation offset when combining
            sepstr += "sep"+str(sep)+"_"
        elif sepsnk:
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
