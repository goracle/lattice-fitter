#!/usr/bin/python

#change file name
import read_file as rf
import numpy as np
import os.path
import avg_mass as am
from os import listdir
from os.path import isfile, join

def getSummed(filen): 
    Lt = sum(1 for line in open(filen))
    ar = np.zeros(shape=(Lt),dtype=complex)
    fn = open(filen, 'r')
    for line in fn:
        l=line.split()
        t=int(l[0])
        ar.itemset(t,complex(l[1]))
    return ar

def proc_and_fix_norm(fn):
    fig = getSummed(fn)
    name = rf.figure(fn)
    vp = rf.vecp(fn)
    norm = 1.0
    if(name == 'R'):
        norm = 4.0
    elif(name == 'T'):
        norm = 2.0
    elif(name == 'C'):
        norm = 2.0
    elif(name == 'D'):
        norm = 2.0
    elif(name == 'Hbub'):
        norm = 2.0
    #elif(name == 'scalar-bubble'):
    #do nothing    
    elif(name == 'V'):
        norm = 4.0
    elif(name == 'Cv3'):
        norm = 2.0
    fig *= norm
    return fig

def I0(lb,ub,show=False):
    onlyfiles=[f for f in listdir('.') if isfile(join('.',f))]
    I0sum = []
    count = 0
    p1 = []
    p2 = []
    p3 = []
    for fn in onlyfiles:
        if rf.vecp(fn):
            continue
        print "Processing:", fn
        fig = proc_and_fix_norm(fn)
        name = rf.figure(fn)
        if name == 'R':
            fig *= -3.0/2.0
        elif name == 'C':
            fig *= 1.0/2.0
        elif name == 'D':
            fig *= 1.0
        elif name == 'V':
            fig *= 3.0
        else:
            continue
        if rf.mom(fn):
            if count==0:
                p1,p2,p3=rf.mom(fn)
            p1t,p2t,p3t=rf.mom(fn)
            if p1t != p1 or p2t != p2 or p3t != p3:
                continue
        if count == 0:
            I0sum = fig
            count+=1
        else:
            I0sum += fig
            count += 1
    Vac = getSummed("AvgVdis_mom1000_mom2000")
    #I0avg = I0sum/count-Vac-30000*1e6*np.ones(shape=(len(I0sum)))
    I0avg=I0sum/count
    if len(Vac)>len(I0avg):
        I0avg.resize(len(Vac))
    #I0avg -= 12*Vac
    outfile = "I0avg"
    if p1 and p2 and p3:
        outfile +="_mom1src"+rf.ptostr(p1)+"_mom2src"+rf.ptostr(p2)
        outfile += "_mom1snk"+rf.ptostr(p3)
    rf.write_vec_str(I0avg,outfile)
    return am.fit_this(outfile,lb,ub,show)

def I1(lb,ub,show=False):
    onlyfiles=[f for f in listdir('.') if isfile(join('.',f))]
    I1sum = []
    count = 0
    p1 = []
    p2 = []
    p3 = []
    fig = None
    for fn in onlyfiles:
        if not rf.vecp(fn):
            continue
        print "Processing:", fn
        fig = proc_and_fix_norm(fn)
        fig = np.array(fig)
        name = rf.figure(fn)
        if name == 'R':
            fig *= 1.0
        elif name == 'D':
            fig *= 1.0
        else:
            continue
        if rf.mom(fn):
            if count==0:
                p1,p2,p3=rf.mom(fn)
            p1t,p2t,p3t=rf.mom(fn)
            if p1t != p1 or p2t != p2 or p3t != p3:
                continue
        if count == 0:
            I1sum = fig
        else:
            I1sum += fig
        count += 1
    if fig == None:
        return 'ERR','ERR'
    I1avg = I1sum/count
    outfile = "I1avg_mom1src"+rf.ptostr(p1)+"_mom2src"+rf.ptostr(p2)
    outfile += "_mom1snk"+rf.ptostr(p3)
    rf.write_vec_str(I1avg,outfile)
    return am.fit_this(outfile,lb,ub,show)

def I2(lb,ub,show=False):
    onlyfiles=[f for f in listdir('.') if isfile(join('.',f))]
    I2sum = []
    count = 0
    p1 = []
    p2 = []
    p3 = []
    for fn in onlyfiles:
        print "Processing:", fn
        if rf.vecp(fn):
            continue
        fig = proc_and_fix_norm(fn)
        name = rf.figure(fn)
        if name == 'C':
            fig *= 1.0
        elif name == 'D':
            fig *= 1.0
        else:
            continue
        if rf.mom(fn):
            if count==0:
                p1,p2,p3=rf.mom(fn)
            p1t,p2t,p3t=rf.mom(fn)
            if p1t != p1 or p2t != p2 or p3t != p3:
                continue
        if count == 0:
            I2sum = fig
        else:
            I2sum += fig
        count += 1
    I2avg = I2sum/count
    outfile = "I2avg"
    if p1 and p2 and p3:
        outfile +="_mom1src"+rf.ptostr(p1)+"_mom2src"+rf.ptostr(p2)
        outfile += "_mom1snk"+rf.ptostr(p3)
    rf.write_vec_str(I2avg,outfile)
    return am.fit_this(outfile,lb,ub,show)

def main():
    show = True
    lb = 4
    ub = 15
    A,E = I0(7,10,show)
    print "Isospin 0, mass = ", E
    A,E = I1(lb,ub,show)
    print "Isospin 1, mass = ", E
    A,E = I2(lb,ub,show)
    print "Isospin 2, mass = ", E

if __name__ == "__main__":
    main()
