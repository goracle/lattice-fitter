#!/usr/bin/python3

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
    tlist = traj_list(onlyfiles)
    lookup = {}
    for traj in tlist:
        disfiles=[]
        for fn in onlyfiles:
            if rf.traj(fn) != traj:
                continue
            fign = rf.figure(fn)
            if not fign in ["scalar-bubble","Vdis"]:
                continue
            disfiles.append(fn)
        for dsrc in disfiles:
            for dsnk in disfiles:
                momsrc = rf.mom(dsrc)
                momsnk = rf.mom(dsnk)
                mt1 = momtotal(momsrc)
                mt2 = momtotal(momsnk)
                if not np.array_equal(mt1,mt2):
                    continue
                momstr = dismom(momsrc,momsnk)
                outfig = comb_fig(dsrc,dsnk)
                if not outfig:
                    continue
                sep1 = rf.sep(dsrc)
                sep2 = rf.sep(dsnk)
                if outfig == 'V' and sep1 != sep2:
                        continue
                if sep1:
                    sep = sep1
                    sepVal=int(sep)
                    sepstr = "_sep"+str(sep)+"_"
                elif sep2:
                    sep = sep2
                    sepVal=int(sep)
                    sepstr = "_sep"+str(sep)+"_"
                else:
                    sep = None
                    sepVal = 0
                    sepstr = "_"
                outfile = "traj_"+str(traj)+"_Figure"+outfig+sepstr+momstr
                if(os.path.isfile(outfile)):
                    print("Skipping:", outfile)
                    print("File exists.")
                    continue
                #get the data
                #Note:  cb.comb_dis defaults to taking the complex conjugate of src only.
                arrPlus = np.array(cb.comb_dis(dsrc,dsnk,sepVal))
                srcFig=rf.figure(dsrc)
                snkFig=rf.figure(dsnk)
                dsrcSub = re.sub(srcFig,"Avg_"+srcFig,d+dsrc)
                dsnkSub = re.sub(snkFig,"Avg_"+snkFig,d+dsnk)
                dsrcSub = re.sub('traj_(\d)+_Figure_','',dsrcSub)
                dsnkSub = re.sub('traj_(\d)+_Figure_','',dsnkSub)
                #get the  <><> subtraction array (<> indicates avg over trajectories)
                if dsrcSub+dsnkSub in lookup:
                    print("Using prev.")
                    arrMinus=lookup[dsrcSub+dsnkSub]
                else:
                    arrMinus = np.array(cb.comb_dis(dsrcSub,dsnkSub,sepVal))
                    lookup[dsrcSub+dsnkSub]=arrMinus
                #arr = arrPlus - arrMinus
                #rf.write_arr(arrPlus - arrMinus,outfile)
                rf.write_arr(arrPlus,outfile)

#to test below this line
#helper function; builds string corresponding to file
def outf(conf,figname,vec,sep,parr,pol1=None, pol2=None):
    v=""
    if(vec):
        v="_vec"
    base="traj_"+str(conf)+"_Figure"+str(figname)+v
    if not pol2 and pol1:
        base+="_pol_snk_"+str(pol1)
        base2=base+"_sep"+str(sep)
    elif pol2 and pol1:
        base+="_pol_src-snk_"+str(pol1)+"-"+str(pol2)
    else:
        base2=base+"_sep"+str(sep)
    l=len(parr)
    one=(type(parr[0]) is int)
    if not one and l==3:
        return base2+"_mom1src"+ptostr(parr[0])+"_mom2src"+ptostr(parr[1])+"_mom1snk"+ptostr(parr[2])
    elif not one and l==2:
        return base2+"_momsrc"+ptostr(parr[0])+"_momsnk"+ptostr(parr[1])
    elif one and l==3:
        return base+"_mom"+ptostr(parr[0])
    else:
        print("Error: bad output filename specification.")
        exit(1)

if __name__ == "__main__":
    main()
