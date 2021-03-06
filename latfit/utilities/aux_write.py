#!/usr/bin/python3
"""Write auxiliary diagrams from files in current directory."""
import sys
from os import listdir
from os.path import isfile, join
import os.path
import numpy as np
import latfit.utilities.read_file as rf
#from latfit.analysis.profile import PROFILE

def transtime(tsrc, tdis, tsep, nmomaux, len_t):
    """Transform tsrc and tdis to aux diagram versions"""
    if nmomaux == 3:
        tsrc2 = (tsrc+tdis+tsep) % len_t
        tdis2 = (3*len_t-2*tsep-tdis) % len_t
    elif nmomaux == 2:
        tsrc2 = (tdis+tsrc) % len_t
        tdis2 = (2*len_t-tsep-tdis) % len_t
    elif nmomaux == 1:
        tsrc2 = (tdis+tsrc) % len_t
        tdis2 = (len_t-tdis) % len_t
    else:
        print("Error: bad filename, error in getaux_filestrs")
        sys.exit(1)
    return tsrc2, tdis2


def getaux_filestrs(filename):
    """gets the array from the file, but keeps the values as strings
    """
    tsep = rf.pion_sep(filename)
    nmomaux = rf.nmom(filename)
    len_t = rf.find_dim(filename)
    out = np.zeros((len_t, len_t), dtype=np.object)
    filen = open(filename, 'r')
    for line in filen:
        lsp = line.split(' ')  # lsp[0] = tsrc, lsp[1] = tdis
        tsrc = int(lsp[0])
        tdis = int(lsp[1])
        tsrc2, tdis2 = transtime(tsrc, tdis, tsep, nmomaux, len_t)
        out[tsrc2][tdis2] = str(lsp[2])+" "+str(lsp[3]).rstrip()
    return out


def ascii_test(filename, stype):
    """Test if a file has a disconnected diagram
    if we are using ascii file format
    """
    if stype == 'ascii':
        print("Processing:", filename)
        if not rf.discon_test(filename):
            filename = None
    return filename

def setup_aux_filen(filename, stype):
    """initial setup for the function aux_filen"""
    filename = ascii_test(filename, stype)
    pret = np.array(rf.mom(filename))
    nmom = rf.nmom(filename)
    plist = nmom*[0]
    outfile = filename
    return outfile, plist, nmom, pret

def outfile_nmom3(pret, plist, outfile, filename, stype):
    """Get output file name for a single momentum"""
    psrc1 = pret[0]
    psrc2 = pret[1]
    psnk1 = pret[2]
    plist[0] = psrc1+psrc2-psnk1
    plist[1] = psnk1
    plist[2] = psrc2
    outfile = rf.pchange(outfile, -1*np.array(plist))  # cc
    if outfile == filename:
        if stype == 'ascii':
            print("symmetric Momenta; skipping")
        outfile = None
    return outfile

def aux_filen(filename, stype='ascii'):
    """Write aux diagram corresponding to filename"""
    outfile, plist, nmom, pret = setup_aux_filen(filename, stype)
    if nmom == 3:
        outfile = outfile_nmom3(pret, plist, outfile, filename, stype)
    elif nmom == 2:
        if 'pol_snk' in outfile or 'scalar_' in outfile:
            psrc1 = pret[0]
            psnk = pret[1]
            psrc2 = psnk-psrc1
            plist[0] = psnk # new source (-1)
            plist[1] = psrc2 # new sink (-1)
        elif 'pol_src' in outfile or 'scalarR_' in outfile:
            psrc = pret[0]
            psnk1 = pret[1]
            psnk2 = psrc-psnk1
            plist[1] = psrc # new sink (-1)
            plist[0] = psnk2 # new source (-1)
        else:
            assert None, "bad filename to apply aux to: "+str(filename)
        outfile = outfile.replace("pol_snk", "pol_SOURCE")
        outfile = outfile.replace("pol_src", "pol_SINK")
        outfile = outfile.replace("pol_SOURCE", "pol_src")
        outfile = outfile.replace("pol_SINK", "pol_snk")
        outfile = outfile.replace("scalar_", "scalarRR_")
        outfile = outfile.replace("scalarR_", "scalar_")
        outfile = outfile.replace("scalarRR_", "scalarR_")
        if outfile == filename:
            if stype == 'ascii':
                print("T aux diagram already exists. Skipping.")
            # outfile = filename.replace("pol_src", "pol_snk")
            # or outfile = filename.replace("scalarR_", "scalar_")
            # print "i.e.", outfile
            outfile = None
        outfile = rf.pchange(outfile, -1*np.array(plist))  # cc
    elif nmom == 1:
        # outfile = outfile.replace("scalar_", "scalarR_")
        outfile = rf.pchange(outfile, -1*pret)  # cc
        if rf.vecp(outfile):
            # swap the polarizations
            pol1, pol2 = rf.pol(outfile)
            btemp = "pol_src-snk_"
            outfile = outfile.replace(btemp+str(pol1), btemp+"temp1")
            outfile = outfile.replace(btemp+"temp1-"+str(
                pol2), btemp+"temp1-temp2")
            outfile = outfile.replace("temp1", str(pol2))
            outfile = outfile.replace("temp2", str(pol1))
        if outfile == filename:
            if stype == 'ascii':
                print("Two point not a vector diagram.  Skipping.")
            outfile = None
    else:
        print("Error: nmom does not equal 1, 2, or 3.")
        sys.exit(1)
    return outfile


def main():
    """Aux write, main"""
    dur = 'aux_diagrams'
    if not os.path.isdir(dur):
        os.makedirs(dur)
    onlyfiles = [f for f in listdir('.') if isfile(join('.', f))]
    for filen in onlyfiles:
        retfn = aux_filen(filen)
        if not retfn:
            continue
        outfile = "aux_diagrams/"+retfn
        if os.path.isfile(outfile):
            print("Skipping:'"+outfile+"'.  File exists.")
            continue
        outarr = getaux_filestrs(filen)
        rf.write_mat_str(outarr, outfile)

    print("Done writing auxiliary periodic bc files")
    sys.exit(0)


if __name__ == "__main__":
    main()
