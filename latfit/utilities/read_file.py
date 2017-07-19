#!/usr/bin/python3
"""General utility functions"""
#to get files
import sys
#to process file list
import os.path
from math import sqrt
import re
import warnings
import numpy as np

def pol(filename):
    """Get polarization info from filename"""
    mat = re.search(r'pol_snk_(\d)', filename)
    nmat = re.search(r'pol_src-snk_(\d)-(\d)', filename)
    if mat:
        polret = int(mat.group(1))
    elif nmat:
        polret = int(nmat.group(1)), int(nmat.group(2))
    else:
        polret = None
    return polret

def traj(filename):
    """Get trajectory info from filename"""
    filename = str(filename)
    mat = re.search('traj_([B0-9]+)_[A-Za-z]{1}', filename)
    if mat:
        rettraj = mat.group(1)
    else:
        warnings.warn(
            "Warning: filename:'"+filename+"', has no configuration info")
        rettraj = None
    return rettraj

def reverse_p(filename):
    """determine if we are dealing with a reverse diagram
    """
    if re.search('scalarR', filename):
        boolp = True
    elif figure(filename) == 'T' and re.search('pol_src', filename):
        boolp = True
    elif figure(filename) == 'Cv3R':
        boolp = True
    else:
        boolp = False
    return boolp

def checkp(filename):
    """determine if file contains check data on vector diagram
    """
    return bool(re.search('_vecCheck_', filename))

def vecp(filename):
    """is the diagram a vector
    """
    return bool(re.search('_vec_', filename))

def figure(filename):
    """return figure name
    """
    hmat = re.search('([A-Za-z0-9]*)corr', filename)
    lmat = re.search('Figure(.*?)$', filename)
    kmat = re.search('Figure_(.*?)$', filename)
    mat = re.search('Figure(.*?)_', filename)
    nmat = re.search('Figure_(.*?)_', filename)
    if nmat:#don't reverse m and n order, helps for some reason
        fign = nmat.group(1)
    elif mat:
        fign = mat.group(1)
    elif lmat:
        fign = lmat.group(1)
    elif kmat:
        fign = kmat.group(1)
    #just a two point single particle correlator
    elif hmat:
        fign = hmat.group(0)
    else:
        warnings.warn("Warning: bad filename, no figure name: "+filename)
        fign = None
    return fign

#how many momenta returned?
def nmom_arr(pret):
    """Get number of momenta from momentum p array"""
    nmom1 = len(pret)
    if isinstance(pret[0], int) and nmom1 == 3:
        return 1
    elif not isinstance(pret[0], int):
        return nmom1
    else:
        print("Error: bad momentum container.")
        sys.exit(1)

def nmom(filename):
    """Get number of momenta from filename"""
    return nmom_arr(mom(filename))

def procmom(mstr):
    """help function for mom(filename).
    gets int momentum array from regex match
    """
    bmat = re.findall(r'_*\d', mstr)
    if len(bmat) != 3:
        print("Error: bad filename, number of momenta")
        sys.exit(1)
    return [int(bmat[i].replace('_', '-')) for i in range(len(bmat))]

def getmomstr(filename):
    """get momentum string from filename"""
    mat = re.search('mom.*', filename)
    if mat:
        pstr = mat.group(0)
    else:
        warnings.warn("Error: no momentum in filename: "+filename)
        pstr = None
    return pstr

def mom(filename):
    """returns momentum of filename in the form of an int array
    """
    kmat = re.search(r'mom1((_?\d+){3})', filename)
    lmat = re.search(r'momsrc((_?\d+){3})', filename)
    mat = re.search(r'mom1src((_?\d+){3})', filename)
    nmat = re.search(r'mom((_?\d+){3})', filename)
    if kmat:
        mom1 = procmom(kmat.group(1))
        kmat = re.search(r'mom2((_?\d+){3})', filename)
        if kmat:
            mom2 = procmom(kmat.group(1))
            pret = (mom1, mom2)
        else:
            print("Error: bad filename, mom2 not found", filename)
            sys.exit(1)
    elif lmat:
        psrc = procmom(lmat.group(1))
        lmat = re.search(r'momsnk((_?\d+){3})', filename)
        if lmat:
            psnk = procmom(lmat.group(1))
            pret = (psrc, psnk)
        else:
            print("Error: bad filename, psnk not found", filename)
            sys.exit(1)
    elif mat:
        pret = three_mat(mat, filename)
    elif nmat:
        pret = procmom(nmat.group(1))
    else:
        print("Error: bad filename= '"+str(filename)+"' no momenta found.  Attempting to continue.")
        pret = None
    return pret
        #sys.exit(1)

def three_mat(mat, filename):
    """get three 3 momenta from filename; called by mom(filename)"""
    psrc1 = procmom(mat.group(1))
    mat = re.search(r'mom2src((_?\d+){3})', filename)
    if mat:
        psrc2 = procmom(mat.group(1))
        mat = re.search(r'mom1snk((_?\d+){3})', filename)
        if mat:
            psnk1 = procmom(mat.group(1))
            pret = (psrc1, psrc2, psnk1)
        else:
            print("Error: bad filename, psnk1 not found", filename)
            sys.exit(1)
    else:
        print("Error: bad filename, psrc2 not found.", filename)
        sys.exit(1)
    return pret

def pion_sep(filename):
    """Get sep info from filename

    returns pion separation (not to be confused with distance)
    i.e. tdis is the separation in time between src/snk
    pion separation is how far apart the pions are in time but mostly localized to one time slice
    """
    mat = re.search(r'_sep(\d+)_', filename)
    if mat:
        sep1 = int(mat.group(1))
    else:
        sep1 = None
    return sep1

def sep(filename):
    """Get t separation, two particles, alias"""
    return pion_sep(filename)

def sum_rows(inmat, avg=False):
    """np.fsum over t_src
    """
    ncol = int(sqrt(inmat.size))
    if avg:
        col = [np.sum(inmat.item(i, j) for i in range(ncol))/(ncol) for j in range(ncol)]
    else:
        col = [np.sum(inmat.item(i, j) for i in range(ncol)) for j in range(ncol)]
    return col

def find_dim(filename):
    """get dimensions of the matrix from the sqrt(num_rows) of the file
    """
    ndim = sum(1 for line in open(filename))
    if sqrt(ndim) == int(sqrt(ndim)):
        len_t = int(sqrt(ndim))
    else:
        print("Error: Non-square matrix, ndim = ", ndim)
        len_t = None
    return len_t

def write_vec_str(data, outfile):
    """write vector of strings
    """
    if os.path.isfile(outfile):
        print("Skipping:", outfile)
        print("File exists.")
        return
    filen = open(outfile, 'w')
    len_t = len(data)
    for tdis in range(len_t):
        cnum = data[tdis]
        if not isinstance(cnum, str):
            cnum = '{0:.{1}f}'.format(cnum, sys.float_info.dig)
        line = str(tdis)+" "+str(cnum)+"\n"
        filen.write(line)
    print("Done writing file:", outfile)
    filen.close()

def write_mat_str(data, outfile):
    """write matrix of strings
    """
    if os.path.isfile(outfile):
        print("Skipping:", outfile)
        print("File exists.")
        return
    filen = open(outfile, 'w')
    len_t = len(data[0])
    if len_t != len(data):
        print("Error: non-square matrix.")
        sys.exit(1)
    for tsrc in range(len_t):
        for tdis in range(len_t):
            cnum = data[tsrc][tdis]
            if not isinstance(cnum, str):
                cnum = '{0:.{1}f}'.format(cnum, sys.float_info.dig)
            line = str(tsrc)+" "+str(tdis)+" "+str(cnum)+"\n"
            filen.write(line)
    print("Done writing file:", outfile)
    filen.close()

def write_arr(data, outfile):
    """write built array to file (for use in building disconnected diagrams)
    """
    if os.path.isfile(outfile):
        print("Skipping:", outfile)
        print("File exists.")
        return
    filen = open(outfile, 'w')
    len_t = len(data[0])
    if len_t != len(data):
        print("Error: non-square matrix. outfile = ", outfile)
        sys.exit(1)
    for tsrc in range(len_t):
        for tdis in range(len_t):
            cnum = data[tsrc][tdis]
            if not isinstance(cnum, str):
                cnum = complex('{0:.{1}f}'.format(cnum, sys.float_info.dig))
            line = str(tsrc)+" "+str(tdis)+" "+str(cnum.real)+" "+str(
                cnum.imag)+"\n"
            filen.write(line)
    print("Done writing file:", outfile)
    filen.close()

def write_block(block, outfile, already_checked=False):
    """write two columns of floats.  (real and imag parts of jackknife block)
    """
    if not already_checked:
        if os.path.isfile(outfile):
            print("Skipping:", outfile)
            print("File exists.")
            return
    with open(outfile, "a") as myfile:
        for line in block:
            if not isinstance(line, str):
                line = complex('{0:.{1}f}'.format(line, sys.float_info.dig))
                line = str(line.real)+" "+str(line.imag)+'\n'
            myfile.write(line)

def ptostr(ploc):
    """makes momentum array into a string
    """
    return (str(ploc[0])+str(ploc[1])+str(ploc[2])).replace("-", "_")

def pchange(filename, pnew):
    """replace momenta in filename
    """
    pold = mom(filename)
    nold = nmom_arr(pold)
    nnew = nmom_arr(pnew)
    traj1 = traj(filename)
    filen = re.sub("traj_.*?_", "traj_TEMPsafe_", filename)
    if nnew != nold:
        print("Error: filename momentum mismatch")
        sys.exit(1)
    nmom1 = nold
    if nmom1 == 1:
        filen = filen.replace(ptostr(pold), ptostr(pnew))
    elif nmom1 == 2 or nmom1 == 3:
        for i in range(nmom1):
            filen = filen.replace(ptostr(pold[i]), "temp"+str(i), 1)
        for i in range(nmom1):
            filen = filen.replace("temp"+str(i), ptostr(pnew[i]), 1)
    else:
        print("Error: bad filename for momentum replacement specified.")
        sys.exit(1)
    filen = re.sub("TEMPsafe", str(traj1), filen)
    return filen

def remp(mom1, mom2, mom3=(0, 0, 0)):
    """helper function; find remaining momentum
    """
    return np.array(mom1)+np.array(mom2)-np.array(mom3)

if __name__ == '__main__':
    pass
