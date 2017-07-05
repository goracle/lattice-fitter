#!/usr/bin/python3

import read_file as rf
import os
import numpy as np
import re
from os import listdir
import os.path
from os.path import isfile, join
import warnings
from math import sqrt
import sys

def sum_blks(outdir,coeffs_arr):
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError:
            print("can't create directory:",outdir,"Permission denied.")
            sys.exit(1)
    print("Start of blocks:",outdir,"--------------------------------------------")
    onlyfiles=[f for f in listdir('./'+coeffs_arr[0][0]) if isfile(join('./'+coeffs_arr[0][0],f))]            
    #make new directory if it doesn't exist
    #loop over time slices until there are none left
    sent=object()
    flag=sent
    for time in onlyfiles:
        if re.search('pdf',time):
            print("Skipping 'block' = ",time)
            continue
        outfile=outdir+'/'+time
        outblk=np.array([])
        for pair in coeffs_arr:
            name,coeff=pair
            if flag == sent:
                print("Including:",name,"with coefficient",coeff)
            #do the check after printing out the coefficients so we can check afterwards
            if(os.path.isfile(outfile)):
                print("Skipping:", outfile)
                print("File exists.")
                continue
            try:
                fn=open(name+'/'+time,'r')
            except:
                print("Error: bad block name in:", name)
                print("block name:",time,"Continuing.")
                continue
            for i,line in enumerate(fn):
                line=line.split()
                if len(line)==1:
                    val=coeff*float(line[0])
                elif len(line)==2:
                    val=coeff*complex(float(line[0]),float(line[1]))
                else:
                    print("Error: bad block:",fn)
                    break
                try:
                    outblk[i]+=val
                except:
                    outblk=np.append(outblk,val)
        flag = 0
        if(os.path.isfile(outfile)):
            continue
        with open(outfile,'a') as fn:
            for line in outblk:
                outline=complex('{0:.{1}f}'.format(line,sys.float_info.dig))
                if outline.imag == 0:
                    outline = str(outline.real)+"\n"
                else:
                    outline = str(outline.real)+" "+str(outline.imag)+"\n"
                fn.write(outline)
            print("Done writing:",outfile)
    print("End of blocks:",outdir,"--------------------------------------------")
                
def norm_fix(fn):
    name = rf.figure(fn)
    norm = 1.0
    if(name == 'R'):
        norm = 1.0
    elif(name == 'T'):
        if rf.vecp(fn) and not rf.reverseP(fn):
            norm = -1.0
        else:
            norm = 1.0
    elif(name == 'C'):
        norm = 1.0
    elif(name == 'D'):
        norm = 1.0
    elif(name == 'Hbub' or name == 'pioncorr'):
        norm = 2.0
    elif(name == 'scalar-bubble'):
        norm = 1.0
    elif(name == 'V'):
        norm = 4.0
    elif(name == 'Cv3'):
        norm = 2.0
    elif(name == 'Cv3R'):
        norm = 2.0
    return norm

def isospin_coeff(fn,I):
    name = rf.figure(fn)
    norm = 1.0
    vecp = rf.vecp(fn)
    if I == 0:
        if(name == 'V'):
            norm = 3.0
        elif(name == 'D' and not vecp):
            norm = 2.0
        elif(name == 'R' and not vecp):
            norm = -6.0
        elif(name == 'C'):
            norm = 1.0
        elif name == 'Cv3' or name == 'Cv3R':
            norm = 5.0/sqrt(6.0)
        elif name == 'T' and not vecp:
            norm = -5.0/(sqrt(6.0))
        elif name == 'Hbub' and not vecp:
            norm = -1.0
        elif name == 'Bub2' or name == 'bub2':
            norm = 2.0
        else:
            return None
    elif I == 1:
        if (name == 'Hbub' and vecp) or name == 'pioncorr':
            norm = 1.0
        elif name == 'T' and vecp:
            norm = 4.0/sqrt(2.0)
        elif name == 'R' and vecp:
            norm = 4.0
        elif name == 'D' and vecp:
            norm = 2.0
        else:
            return None
    elif I == 2:
        if name == 'D' and not vecp:
            norm = 2.0
        elif name == 'C':
            norm = -2.0
        else:
            return None
    else:
        print("Error: bad isospin:", I)
        exit(1)
    return norm

def momtotal(pl,fig=None):
    if len(pl)==3 and type(pl[0]) is int:
        return pl
    elif len(pl)==2:
        m=re.search('scalarR',fig)
        n=re.search('pol_src',fig)
        if m or n:
            return pl[0]
        else:
            return pl[1]
    elif len(pl)==3 and not type(pl[0]) is int:
        p1=np.array(pl[0])
        p2=np.array(pl[1])
        ptotal=list(p1+p2)
        return ptotal
    else:
        print("Error: bad momentum list:",pl)
        exit(1)

def main(fixn,DIRNUM):
    d='.'
    dlist=[os.path.join(d,o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    seplist=set()
    momlist=set()
    pion=set(['pioncorr'])
    pipi=set(['C','D','R','V'])
    pipirho=set(['T'])
    rhopipi=set(['T'])
    pipisigma=set(['Cv3R','T'])
    sigmapipi=set(['Cv3','T'])
    sigmasigma=set(['Hbub','Bub2','bub2'])
    rhorho=set(['Hbub'])
    #def is a tuple consisting of list of particles, Isospin, and whether the diagram is a reverse diagram
    filterlist={'pion':(pion,[1],False),'pipi':(pipi,[0,1,2],False),'pipirho':(pipirho,[1],True),'pipisigma':(pipisigma,[0],True),
                'sigmasigma':(sigmasigma,[0],False),'rhorho':(rhorho,[1],False),'sigmapipi':(sigmapipi,[0],False),'rhopipi':(rhopipi,[1],False)}
    for d in dlist:
        seplist.add(rf.sep(d))
        momlist.add(rf.getmomstr(d))
        #mom1=rf.mom(d)
        #if not mom1:
        #    continue
        #if len(mom1)==2:
        #    momlist.add(tuple(momtotal(mom1,rf.figure(d))))
        #else:
        #    momlist.add(tuple(momtotal(mom1)))
    for op in filterlist:
        for I in filterlist[op][1]:
            for sep in seplist:
                for mom in momlist:
                    #mom=list(mom)
                    coeffs_arr = []
                    for d in dlist:
                        #if momtotal(rf.mom(d),d) != mom:
                        if rf.getmomstr(d) != mom:
                            continue
                        if not rf.figure(d) in filterlist[op][0]:
                            continue
                        if rf.sep(d) != sep:
                            continue
                        if rf.reverseP(d) is not filterlist[op][2]:
                            continue
                        if re.search('Check',d) or re.search('Chk',d) or re.search('chk',d) or re.search('check',d):
                            continue
                        norm1=isospin_coeff(d,I)
                        if not norm1:
                            continue
                        if fixn:
                            norm2=norm_fix(d)
                        else:
                            norm2=1.0
                        norm=norm1*norm2
                        coeffs_arr.append((d,norm))
                    if coeffs_arr == []:
                        continue
                    if DIRNUM == 0:
                        sepstr=''
                        if sep:
                            sepstr=sepstr+"sep"+str(sep)+'/'
                        #outdir = op+"_I"+str(I)+sepstr+mom
                        outdir = 'I'+str(I)+'/'+sepstr+op+'_'+mom
                    elif DIRNUM == 1:
                        sepstr='_'
                        if sep:
                            sepstr=sepstr+"sep"+str(sep)+'_'
                        #outdir = op+"_I"+str(I)+sepstr+"_momtotal"+rf.ptostr(mom)
                        outdir = op+"_I"+str(I)+sepstr+mom
                    else:
                        print("Error: bad flag specified. DIRNUM =", DIRNUM)
                        sys.exit(1)
                    sum_blks(outdir,coeffs_arr)
    print("Done writing jackknife sums.")
                    
    #
if __name__ == '__main__':
    fixn=input("Need fix norms before summing? True/False?")
    if fixn=='True':
        fixn=True
    elif fixn=='False':
        fixn=False
    else:
        sys.exit(1)
    main(fixn,0)
    main(fixn,1)
