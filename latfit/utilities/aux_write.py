#!/usr/bin/python3

import read_file as rf
import re
import os.path
from os import listdir
from os.path import isfile,join

#gets the array from the file, but keeps the values as strings
def proc_file_str(filename):
    Lt = rf.find_dim(filename)
    tsep=rf.pion_sep(filename)
    n=rf.nmom(filename)
    out=np.zeros((Lt,Lt),dtype=np.object)
    fn=open(filename,'r')
    for line in fn:
        l=line.split(' ')#l[0]=tsrc, l[1]=tdis
        tsrc=int(l[0])
        tdis=int(l[1])
        if n==3:
            tsrc2=(tsrc+tdis+tsep)%Lt
            tdis2=(3*Lt-2*tsep-tdis)%Lt
        elif n==2:
            tsrc2=(tdis+tsrc)%Lt
            tdis2=(2*Lt-tsep-tdis)%Lt
        elif n==1:
            tsrc2=(tdis+tsrc)%Lt
            tdis2=(Lt-tdis)%Lt
        else:
            print("Error: bad filename, error in proc_file_str")
            exit(1)
        out[tsrc2][tdis2]=str(l[2])+" "+str(l[3]).rstrip()
    return out

def call_aux(fn,out):
    if(os.path.isfile(out)):
        print("Skipping:'"+out+"'.  File exists.")
    else:
        rf.write_mat_str(proc_file_str(fn),out)
    return
    

def aux_fn(filename):
    print("Processing:", filename)
    fn=open(filename,'r')
    for line in fn:
        if len(line.split())<4:
            print("Disconnected.  Skipping.")
            return
    pret=np.array(rf.mom(filename))
    n=rf.nmom(filename)
    pn=n*[0]
    outfile=filename
    if n==3:
        psrc1=pret[0]
        psrc2=pret[1]
        psnk1=pret[2]
        psnk2=psrc1+psrc2-psnk1
        pn[0]=psnk2
        pn[1]=psnk1
        pn[2]=psrc2
        outfile=rf.pchange(outfile,pn)
        if(outfile==filename):
            print("symmetric Momenta; skipping")
            return
    elif n==2:
        psrc1=pret[0]
        psnk=pret[1]
        psrc2=psnk-psrc1
        pn[0]=psnk
        pn[1]=psrc2
        outfile=outfile.replace("pol_snk","pol_SOURCE")
        outfile=outfile.replace("pol_src","pol_SINK")
        outfile=outfile.replace("pol_SOURCE","pol_src")
        outfile=outfile.replace("pol_SINK","pol_snk")
        outfile=outfile.replace("scalar_","scalarR_")
        if outfile == filename:
            print("T aux diagram already exists. Skipping.")
            #outfile=filename.replace("pol_src","pol_snk")
            #or outfile=filename.replace("scalarR_","scalar_")
            #print "i.e.", outfile
            return
        outfile=rf.pchange(outfile,pn)
    elif n==1:
        #outfile=outfile.replace("scalar_","scalarR_")
        if rf.vecp(outfile):
            #swap the polarizations
            pol1,pol2=rf.pol(outfile)
            b="pol_src-snk_"
            outfile=outfile.replace(b+str(pol1),b+"temp1")
            outfile=outfile.replace(b+"temp1-"+str(pol2),b+"temp1-temp2")
            outfile=outfile.replace("temp1",str(pol2))
            outfile=outfile.replace("temp2",str(pol1))
        if outfile == filename:
            print("Two point not a vector diagram.  Skipping.")
            return
    else:
        "Error: nmom does not equal 1,2, or 3."
        exit(1)
    return call_aux(filename,"aux_diagrams/"+outfile)

def main():
    d='aux_diagrams'
    if not os.path.isdir(d):
        os.makedirs(d)
    onlyfiles=[f for f in listdir('.') if isfile(join('.',f))]
    for fn in onlyfiles:
        aux_fn(fn)
    print("Done writing auxiliary periodic bc files")
    exit(0)
    
if __name__ == "__main__":
    main()
