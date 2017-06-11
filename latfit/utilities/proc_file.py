#!/usr/bin/python

import read_file as rf
import re
import jk_make as jk
import os.path
from os import listdir
from os.path import isfile, join
import numpy as np
import subtractV as subV
import gc

#gets the array from the file
#also gets the GEVP transpose array
def proc_file(filename, sum_tsrc = True):
    Lt = rf.find_dim(filename)
    if not Lt:
        return None
    front=np.zeros(shape=(Lt,Lt),dtype=complex)
    #back=front
    fn=open(filename,'r')
    for line in fn:
        l=line.split()
        if len(l) != 4:
            return None
        #l[0]=tsrc, l[1]=tdis
        tsrc=int(l[0])
        #tsnk=(int(l[0])+int(l[1]))%Lt
        tdis=int(l[1])
        #tdis2=Lt-int(l[1])-1
        front.itemset(tsrc,tdis,complex(float(l[2]),float(l[3])))
        #back.itemset(tsnk,tdis2,complex(float(l[2]),float(l[3])))
    if sum_tsrc:
        #return sum_rows(front), sum_rows(back)
        return rf.sum_rows(front, True)
    else:
        return front
        #return front,back

def call_sum(fn,d,binsize=1,binNum=1,already_summed=False):
    #get the output file name
    if binsize != 1:
        traj=rf.traj(fn)
        if re.search('traj_(\d+)B(\d+)_',traj):
            print "Skipping. File to process is already binned:",fn
            outfile = None
        else:
            outfile = re.sub(str(traj),str(binsize)+'B'+str(binNum),fn)
    else:
        outfile = fn
    #we've obtained the output file name, now get the data
    if not outfile:
        data = None
    elif os.path.isfile(d+outfile):
        print "Skipping:", fn, "File exists."
        data = None
    else:
        if not already_summed:
            data=proc_file(fn, True)
        else:
            data=subV.procV(fn)
        if type(data) is type(None):
            print "Skipping file", fn, "should be 4 numbers per line, non-4 value found."
    return data,outfile


def bin_tsrc_sum(binsize,step,already_summed=False):
    nmax = int(binsize)/int(step)
    onlyfiles=[f for f in listdir('.') if isfile(join('.',f))]
    if nmax == 1:
        if not already_summed:
            d='summed_tsrc_diagrams/'
            if not os.path.isdir(d):
                os.makedirs(d)
        else:
            return
        for fn in onlyfiles:
            data,outfile = call_sum(fn,d,1)
            if data and outfile:
                rf.write_vec_str(data, d+outfile)
        print "Done writing files averaged over tsrc."
        return
    else:
        if not already_summed:
            d='summed_tsrc_diagrams/binned_diagrams/binsize'+str(binsize)+'/'
        else:
            d='binned_diagrams/binsize'+str(binsize)+'/'
        if not os.path.isdir(d):
            os.makedirs(d)
        of={}
        for f in onlyfiles:
            base=jk.baseN(f)
            try:
                traj=int(rf.traj(f))
            except:
                continue
            #throw out trajectories not thermalized
            if traj<1000:
                continue
            if not base in of:
                of[base]=[(f,rf.traj(f))]
            else:
                of[base].append((f,rf.traj(f)))
        for base in of:
            count = 0
            binNum = 0
            data=None
            print "Processing base",base
            #sort by trajectory number
            blist=np.array(sorted(of[base], key=lambda tup: int(tup[1])))
            for fn in blist[:,0]:
                odata,outfile=call_sum(fn,d,binsize,binNum,already_summed)
                if type(odata) is type(None) or not outfile:
                    continue
                if type(data) is type(None):
                    data=np.array(odata)
                else:
                    data+=np.array(odata)
                print "Processed",fn
                count += 1
                if count % nmax == 0:
                    print "Accumulated data.  Writing binned diagram."
                    data/=nmax
                    rf.write_vec_str(data, d+outfile)
                    binNum += 1
                    count = 0
                    data=None
        print "Done writing files binned with bin size =",binsize
        return

def main():
    #global params, set by hand
    mdstep=int(raw_input("Please enter average non-blocked separation."))
    as1=raw_input("Already summed? y/n")
    if as1 == 'y':
        already_summed=True
    elif as1 == 'n':
        already_summed=False
    else:
        exit(1)
    #end global params
    for binsize in [mdstep,20,40,60,80]:
        if binsize == mdstep:
            if not already_summed:
                print "Averaging diagrams over tsrc."
            else:
                continue
        else:
            print "Doing binsize:",binsize
        bin_tsrc_sum(binsize,mdstep,already_summed)
    
if __name__ == "__main__":
    main()