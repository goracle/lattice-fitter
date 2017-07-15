#!/usr/bin/python3

import sys
import read_file as rf
import os.path
from traj_list import traj_list
import numpy as np
import combine as cb
from os import listdir
from os.path import isfile, join
import re
#gets the array from the file, but keeps the values as strings

def comb_fig(dsrc,dsnk):
    figSrc = rf.figure(dsrc)
    figSnk = rf.figure(dsnk)
    if figSrc == 'scalar-bubble' and figSnk == 'scalar-bubble':
        return 'Bub2'
    elif figSrc == 'scalar-bubble' and figSnk == 'Vdis':
        return 'Cv3'
    elif figSrc == 'Vdis' and figSnk == 'scalar-bubble':
        return 'Cv3R'
    elif figSrc == 'Vdis' and figSnk == 'Vdis':
        return 'V'

def singleP(p):
    if len(p) == 3 and type(p[0]) is int:
        return True
    else:
        return False

def momtotal(mom):
    if singleP(mom):
        return mom
    else:
        p1=np.array(mom[0])
        p2=np.array(mom[1])
        return p1+p2

def dismom(psrc,psnk):
    l = len(psrc)+len(psnk)
    if l == 4:
        #V
        p1src=psrc[0]
        p2src=psrc[1]
        #reverse meaning of inner and outer, so take [1] for inner
        p1snk=psnk[1]
        s = "mom1src"+rf.ptostr(p1src)+"_mom2src"+rf.ptostr(p2src)+"_mom1snk"+rf.ptostr(p1snk)
    elif l == 5:
        #Cv3
        if singleP(psrc):
            momsrc = psrc
            #reverse meaning of inner and outer, so take [1] for inner
            momsnk = psnk[1]
        elif singleP(psnk):
            momsnk = psnk
            momsrc = psrc[0]
        s = "momsrc"+rf.ptostr(momsrc)+"_momsnk"+rf.ptostr(momsnk)
    elif l == 6:
        #Bub2
        s = "mom"+rf.ptostr(psrc)
    else:
        print("Error: bad momenta:",psrc,psnk)
        exit(1)
    return s

def main():
    d='summed_tsrc_diagrams/'
    onlyfiles=[f for f in listdir('.') if isfile(join('.',f))]
    lookup = {}
    file_lookup = {}
    for fn in onlyfiles:
        fign = rf.figure(fn)
        if not fign in ["scalar-bubble","Vdis"]:
            continue
        traj = rf.traj(fn)
        mom = rf.mom(fn)
        file_lookup.setdefault(traj,{}).setdefault(
            rf.ptostr(momtotal(mom)),[]).append((fn,mom))
    for traj in file_lookup:
        for mt1 in file_lookup[traj]: 
            for dsrc, momsrc in file_lookup[traj][mt1]:
                for dsnk, momsnk in file_lookup[traj][mt1]:
                    outfig = comb_fig(dsrc,dsnk)
                    sepsrc = rf.sep(dsrc)
                    sepsnk = rf.sep(dsnk)
                    if outfig == 'V' and sepsrc != sepsnk:
                            continue
                    momstr = dismom(momsrc,momsnk)
                    sepVal=0
                    sep = None
                    sepstr = "_"
                    if sepsrc:
                        sep = sepsrc
                        #sepVal=0
                        #we do this because src pipi bubbles don't
                        #need a separation offset when combining
                        sepstr += "sep"+str(sep)+"_"
                    elif sepsnk:
                        sep = sepsnk
                        sepVal=int(sep)
                        sepstr += "sep"+str(sep)+"_"
                    outfile = "traj_"+str(
                        traj)+"_Figure"+outfig+sepstr+momstr
                    flag = 0
                    if(os.path.isfile(outfile)):
                        print("Skipping:", outfile)
                        print("File exists.")
                        #skip the latter write if so
                        flag = 1
                    else:
                        #get the non vac subtraction data
                        #Note:  cb.comb_dis defaults to
                        #not taking the complex conjugate.
                        arrPlus = np.array(cb.comb_dis(dsrc,dsnk,sepVal))
                    outavg = 'AvgVac_Figure'+outfig+sepstr+momstr
                    if os.path.isfile(outavg):
                        print("Skipping:", outavg)
                        print("File exists.")
                        continue
                    srcFig=rf.figure(dsrc)
                    snkFig=rf.figure(dsnk)
                    dsrcSub = re.sub(srcFig,"Avg_"+srcFig,d+dsrc)
                    dsnkSub = re.sub(snkFig,"Avg_"+snkFig,d+dsnk)
                    dsrcSub = re.sub('traj_(\d)+_Figure_','',dsrcSub)
                    dsnkSub = re.sub('traj_(\d)+_Figure_','',dsnkSub)
                    if not os.path.isfile(dsrcSub) or not os.path.isfile(
                            dsnkSub):
                        print("Please do the bubble averaging first.")
                        print("Missing either",dsrcSub,'or',dsnkSub)
                        sys.exit(1)
                    #get the  <><> subtraction array
                    #(<> indicates avg over trajectories)
                    if dsrcSub+dsnkSub in lookup:
                        print("Using prev.")
                        arrMinus=lookup[dsrcSub+dsnkSub]
                    else:
                        arrMinus = np.array(
                            cb.comb_dis(dsrcSub,dsnkSub,sepVal))
                        if not os.path.isfile(outavg):
                            rf.write_arr(arrMinus, outavg)
                        lookup[dsrcSub+dsnkSub]=arrMinus
                    #arr = arrPlus - arrMinus
                    if flag == 0:
                        rf.write_arr(arrPlus - arrMinus,outfile)
                    #rf.write_arr(arrPlus,outfile)

if __name__ == "__main__":
    main()
