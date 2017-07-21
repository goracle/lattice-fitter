#!/usr/bin/python3
"""Write auxiliary diagrams from files in current directory."""
import sys
import os.path
from os import listdir
from os.path import isfile, join
import numpy as np
import read_file as rf

def call_aux(filen, out):
    """Intermediate function.  test if file; if not, write"""
    if os.path.isfile(out):
        print("Skipping:'"+out+"'.  File exists.")
    else:
        rf.write_mat_str(rf.getaux_filestrs(filen), out)
    return

def aux_filen(filename):
    """Write aux diagram corresponding to filename"""
    print("Processing:", filename)
    if not rf.discon_test(filename):
        return
    pret = np.array(rf.mom(filename))
    nmom = rf.nmom(filename)
    plist = nmom*[0]
    outfile = filename
    if nmom == 3:
        psrc1 = pret[0]
        psrc2 = pret[1]
        psnk1 = pret[2]
        psnk2 = psrc1+psrc2-psnk1
        plist[0] = psnk2
        plist[1] = psnk1
        plist[2] = psrc2
        outfile = rf.pchange(outfile, plist)
        if outfile == filename:
            print("symmetric Momenta; skipping")
            return
    elif nmom == 2:
        psrc1 = pret[0]
        psnk = pret[1]
        psrc2 = psnk-psrc1
        plist[0] = psnk
        plist[1] = psrc2
        outfile = outfile.replace("pol_snk", "pol_SOURCE")
        outfile = outfile.replace("pol_src", "pol_SINK")
        outfile = outfile.replace("pol_SOURCE", "pol_src")
        outfile = outfile.replace("pol_SINK", "pol_snk")
        outfile = outfile.replace("scalar_", "scalarR_")
        if outfile == filename:
            print("T aux diagram already exists. Skipping.")
            #outfile = filename.replace("pol_src", "pol_snk")
            #or outfile = filename.replace("scalarR_", "scalar_")
            #print "i.e.", outfile
            return
        outfile = rf.pchange(outfile, plist)
    elif nmom == 1:
        #outfile = outfile.replace("scalar_", "scalarR_")
        if rf.vecp(outfile):
            #swap the polarizations
            pol1, pol2 = rf.pol(outfile)
            btemp = "pol_src-snk_"
            outfile = outfile.replace(btemp+str(pol1), btemp+"temp1")
            outfile = outfile.replace(btemp+"temp1-"+str(
                pol2), btemp+"temp1-temp2")
            outfile = outfile.replace("temp1", str(pol2))
            outfile = outfile.replace("temp2", str(pol1))
        if outfile == filename:
            print("Two point not a vector diagram.  Skipping.")
            return
    else:
        print("Error: nmom does not equal 1, 2, or 3.")
        sys.exit(1)
    return call_aux(filename, "aux_diagrams/"+outfile)

def main():
    """Aux write, main"""
    dur = 'aux_diagrams'
    if not os.path.isdir(dur):
        os.makedirs(dur)
    onlyfiles = [f for f in listdir('.') if isfile(join('.', f))]
    for filen in onlyfiles:
        aux_filen(filen)
    print("Done writing auxiliary periodic bc files")
    sys.exit(0)

if __name__ == "__main__":
    main()
