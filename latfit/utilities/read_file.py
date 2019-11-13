#!/usr/bin/python3
"""General utility functions"""
import sys
import os.path
from math import sqrt
from collections import Iterable
import re
import warnings
import linecache as lc
import numpy as np
import h5py
from accupy import kdot
from latfit.utilities import exactmean as em

FORMAT = 'ASCII'

# regex on filename stuff

def earliest_time(fname):
    """Find the earliest time in the fit range result"""
    spl = fname.split("_")
    tmin = None
    tmax = None
    tminus = 0
    dtee = 0
    for i in spl:
        try:
            i = float(i)
        except ValueError:
            if 'TMINUS' in i:
                i = i.split('.')[0]
                i = i.split('TMINUS')[-1]
                tminus = int(i)
            elif 'dt' in i:
                i = i.split('.')[0]
                i = i.split('dt')[-1]
                dtee = int(i)
            continue
        if tmin is None:
            tmin = i
        elif tmax is None:
            tmax = i
    dtee = max(dtee, tminus)
    assert tmin is not None, str(fname)
    assert tmax is not None, str(fname)
    ret = tmin-dtee
    assert ret >= 0, str(fname)
    return ret

def pol(filename):
    """Get polarization info from filename"""
    mat = re.search(r'pol_snk_(\d)', filename)
    mat2 = re.search(r'pol_src_(\d)', filename)
    nmat = re.search(r'pol_src-snk_(\d)-(\d)', filename)
    assert not (mat and nmat), "Conflicting polarizations found."
    if mat:
        polret = int(mat.group(1))
    elif mat2:
        polret = int(mat2.group(1))
    elif nmat:
        polret = int(nmat.group(1)), int(nmat.group(2))
    else:
        polret = None
    return polret

def firstpol(filename):
    """Just get the first polarization
    (exists a common case where second == first)"""
    ret = pol(filename)
    if isinstance(ret, int) or ret is None:
        pass
    else:
        ret = int(ret[0])
    return ret


def compare_pols(pol1, pol2):
    """Return true if both polarization specifiers are the same.
    false if they are different
    None if one is None
    """
    if pol1 is None or pol2 is None:
        retval = None
    elif str(pol1).isdigit() and str(pol2).isdigit():
        retval = pol1 == pol2
    elif str(pol1).isdigit():
        retval = all([int(i) == int(pol1) for i in pol2])
    elif str(pol2).isdigit():
        retval = all([int(i) == int(pol2) for i in pol1])
    else:
        retval = all([int(i) == int(j) for i in pol1 for j in pol2])
    assert pol1 is None or pol2 is None or retval is not None,\
        "bug pols:"+str(pol1)+" "+str(pol2)
    return retval

def momrelnorm(mom1, mom2):
    """Find the norm of the relative momentum"""
    ret = np.array(mom1)-np.array(mom2)
    return np.sqrt(kdot(ret, ret))

def compare_momenta(fn1, fn2):
    """Compare two different momenta
    (in list form) for equality"""
    return fn1[0] == fn2[0] and fn1[1] == fn2[1] and fn1[2] == fn2[2]

def mom_prefix(fn1, suffix=None):
    """Get everything preceding _mom in a string"""
    split = fn1.split('_')
    ret = ''
    for substr in split:
        if 'mom' in substr:
            break
        ret = ret + substr + '_'
    if ret:
        ret = ret[:-1]
    suffix = mom_suffix(fn1, ret) if suffix is None else suffix
    assert ret + '_' + suffix == fn1, "bug in mom_(pre)(suf)fix"
    return ret

def mom_suffix(fn1, prefix=None):
    """Get everything proceding (and including) _mom in a string"""
    split = fn1.split('_')
    ret = ''
    for i, substr in enumerate(split):
        if 'mom' not in substr:
            continue
        split = split[i:]
        break
    for substr in split:
        ret = ret + substr + '_'
    if ret:
        ret = ret[:-1]
    prefix = mom_prefix(fn1, ret) if prefix is None else prefix
    assert prefix + '_' + ret == fn1, "bug in mom_(pre)(suf)fix"
    return ret

def norm2(momf):
    """p[0]^2+p[1]^2"+p[2]^2"""
    return momf[0]**2+momf[1]**2+momf[2]**2

def traj(filename, nowarn=False):
    """Get trajectory info from filename"""
    filename = str(filename)
    mat = re.search('traj_([B0-9]+)_[A-Za-z]{1}', filename)
    if mat:
        rettraj = mat.group(1)
    else:
        if not nowarn:
            warnings.warn("Warning: filename:" + filename +
                          " , has no configuration info")
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

def allzero(momf):
    """Check if momentum components are all 0"""
    if isinstance(momf[0], Iterable):
        for onep in momf:
            ret = not np.any(onep)
    else:
        ret = not np.any(momf)
    return ret


def checkp(filename):
    """determine if file contains check data on vector diagram
    """
    return bool(re.search('_vecCheck_', filename))

def kaonp(filename):
    """check if a kaon correlator"""
    return bool(re.search('kaon', filename))

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
    omat = re.search('type(.*?)', filename)
    if nmat:  # don't reverse m and n order, helps for some reason
        fign = nmat.group(1)
    elif mat:
        fign = mat.group(1)
    elif lmat:
        fign = lmat.group(1)
    elif kmat:
        fign = kmat.group(1)
    elif hmat: # just a two point single particle correlator
        fign = hmat.group(0)
    elif omat:
        fign = omat.group(0)
    else:
        warnings.warn("Warning: bad filename, no figure name: "+filename)
        fign = None
    return fign


def basename(filen):
    """get basename of file
    """
    mat = re.search('traj_[B0-9]+_(.*)', filen)
    if not mat:
        return None
    return mat.group(1)


# how many momenta returned?
def nmom_arr(pret):
    """Get number of momenta from momentum p array"""
    nmom1 = len(pret)
    if isinstance(pret[0], (np.integer, int)) and nmom1 == 3:
        retval = 1
    elif not isinstance(pret[0], (np.integer, int)):
        retval = nmom1
    else:
        print("Error: bad momentum container.")
        sys.exit(1)
    return retval


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


def mom(filename, printerr=True):
    """returns momentum of filename in the form of an int array
    """
    kmat = re.search(r'mom1((_?\d+){3})', filename)
    lmat = re.search(r'momsrc((_?\d+){3})', filename)
    mat = re.search(r'mom1src((_?\d+){3})', filename)
    nmat = re.search(r'mom((_?\d+){3})', filename)
    if kmat or lmat:
        pret = two_mat(kmat, lmat, filename)
    elif mat:
        pret = three_mat(mat, filename)
    elif nmat:
        pret = procmom(nmat.group(1))
    else:
        if printerr:
            print("Error: bad filename= '"+str(
                filename)+"' no momenta found.  Attempting to continue.")
        pret = None
    return pret


def two_mat(kmat, lmat, filename):
    """get two 3 momenta from filename; called by mom(filename)"""
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
    return pret


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
    pion separation is how far apart the pions are in time
    but mostly localized to one time slice
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


def pchange(filename, pnew):
    """replace momenta in filename
    """
    pold = mom(filename)
    nold = nmom_arr(pold)
    nnew = nmom_arr(pnew)
    traj1 = traj(filename, nowarn=True)
    filen = re.sub("traj_.*?_", "traj_TEMPsafe_",
                   filename) if traj1 is not None else filename
    if nnew != nold:
        print("Error: filename momentum mismatch")
        sys.exit(1)
    nmom1 = nold
    filen = filen.replace(ptostr(pold), ptostr(pnew)) if nmom1 == 1 else filen
    if nmom1 == 2 or nmom1 == 3:
        for i in range(nmom1):
            filen = filen.replace(ptostr(pold[i]), "temp"+str(i), 1)
        for i in range(nmom1):
            filen = filen.replace("temp"+str(i), ptostr(pnew[i]), 1)
    else:
        if nmom1 != 1:
            print("Error: bad filename for momentum replacement specified.")
            print("filen:", filen)
            print("filename:", filename)
            sys.exit(1)
    filen = re.sub(
        "TEMPsafe", str(traj1), filen) if traj1 is not None else filen
    return filen


def remp(mom1, mom2, mom3=(0, 0, 0)):
    """helper function; find remaining momentum
    """
    return np.array(mom1)+np.array(mom2)-np.array(mom3)


# file io stuff

if FORMAT == 'ASCII':
    # test functions
    def discon_test(filename):
        """Test if diagram is disconnected"""
        filen = open(filename, 'r')
        for line in filen:
            if len(line.split()) < 4:
                print("Disconnected.  Skipping.")
                return False
        return True

    def tryblk(name, time):
        """Try to open a file for some purpose"""
        try:
            filen = open(name+'/'+time, 'r')
        except IOError:
            print("Error: bad block name in:", name)
            print("block name:", time, "Continuing.")
            return False
        return filen

    # read file

    def get_block_data(filen, onlydirs):
        """Get array of jackknife block data (from single time slice file, e.g.)
        """
        retarr = np.array([], dtype=complex)
        filen = open(filen, 'r')
        for line in filen:
            lsp = line.split()
            if len(lsp) == 2:
                retarr = np.append(
                    retarr, complex(float(lsp[0]), float(lsp[1])))
            elif len(lsp) == 1:
                retarr = np.append(retarr, complex(lsp[0]))
            else:
                print("Not a jackknife block.  exiting.")
                print("cause:", filen)
                print(onlydirs)
                sys.exit(1)
        return retarr

    def get_linejk(basename2, time, trajl):
        """Return line from file for jackknife blocking"""
        return np.array([complex(lc.getline("traj_"+str(
            traj)+basename2, time+1).split()[1])
                         for traj in trajl])

    def proc_file(filename, sum_tsrc=True):
        """gets the array from the file, optionally sum tsrc
        """
        len_t = find_dim(filename)
        if not len_t:
            return None
        front = np.zeros(shape=(len_t, len_t), dtype=complex)
        # back = front
        filen = open(filename, 'r')
        for line in filen:
            lsp = line.split()
            if len(lsp) != 4:
                return None
            # lsp[0] = tsrc, lsp[1] = tdis
            tsrc = int(lsp[0])
            # tsnk = (int(lsp[0])+int(lsp[1]))%len_t
            tdis = int(lsp[1])
            # tdis2 = len_t-int(lsp[1])-1
            front.itemset(tsrc, tdis, complex(float(lsp[2]), float(lsp[3])))
            # back.itemset(tsnk, tdis2, complex(float(lsp[2]), float(lsp[3])))
        if sum_tsrc:
            # return sum_rows(front), sum_rows(back)
            retarr = sum_rows(front, True)
        else:
            retarr = front
        return retarr

    def proc_vac_real(filen):
        """Get the bubble from the file
        """
        len_t = sum(1 for line in open(filen))
        retarr = np.zeros(shape=(len_t), dtype=complex)
        filen = open(filen, 'r')
        for line in filen:
            lsp = line.split()
            tsrc = int(lsp[0])
            if len(lsp) == 3:
                retarr.itemset(tsrc, float(lsp[1]))
            elif len(lsp) == 2:
                retarr.itemset(tsrc, complex(lsp[1]).real)
            else:
                print("Not a disconnected or summed diagram.  exiting.")
                print("cause:", filen)
                sys.exit(1)
        return retarr

    def proc_vac(filen):
        """Get the bubble from the file
        """
        len_t = sum(1 for line in open(filen))
        retarr = np.zeros(shape=(len_t), dtype=complex)
        filen = open(filen, 'r')
        for line in filen:
            lsp = line.split()
            tsrc = int(lsp[0])
            if len(lsp) == 3:
                retarr.itemset(tsrc, complex(float(lsp[1]), float(lsp[2])))
            elif len(lsp) == 2:
                retarr.itemset(tsrc, complex(lsp[1]))
            else:
                print("Not a disconnected or summed diagram.  exiting.")
                print("cause:", filen)
                sys.exit(1)
        return retarr

    # test functions which read

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

    def numlines(fn1):
        """Find number of lines in a file"""
        return sum(1 for line in open(fn1))

    # write file

    def write_blk(outblk, outfile, already_checked=False):
        """Write numerical array of jackknife block to file"""
        if not already_checked:
            if os.path.isfile(outfile):
                print("Skipping:", outfile)
                print("File exists.")
                return
        with open(outfile, 'a') as filen:
            for line in outblk:
                if not isinstance(line, str):
                    line = complex(
                        '{0:.{1}f}'.format(line, sys.float_info.dig))
                    line = str(line.real)+" "+str(line.imag)+"\n"
                filen.write(line)
            print("Done writing:", outfile)

    WRITE_BLOCK = write_blk

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
        real part seperated from imag by plus sign;
        also imaginary piece has a 'j'
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
        real part seperated from imag by space
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
                    cnum = complex(
                        '{0:.{1}f}'.format(cnum, sys.float_info.dig))
                line = str(tsrc)+" "+str(tdis)+" "+str(cnum.real)+" "+str(
                    cnum.imag)+"\n"
                filen.write(line)
        print("Done writing file:", outfile)
        filen.close()

elif FORMAT == 'HDF5':
    # test functions
    def discon_test(filename):
        """Test if disconnected diagram (hdf5)"""
        filenp = h5py.File(filename, 'r')
        filen = filenp[filename]
        dimf = len(filen)
        if filen.shape != (dimf, dimf):
            print("Disconnected.  Skipping.")
            return False
        return True
    # to do, maybe

    def tryblk(name, time):
        """Try to open file for some purpose (hdf5)"""
        try:
            filen = open(name+'/'+time, 'r')
        except IOError:
            print("Error: bad block name in:", name)
            print("block name:", time, "Continuing.")
            return False
        return filen
    # write file

    def write_blk(outblk, outfile, already_checked=False):
        """Write numerical array of jackknife block to file"""
        if not already_checked:
            if os.path.isfile(outfile):
                print("Skipping:", outfile)
                print("File exists.")
                return
        outf = h5py.File(outfile, 'w')
        outf[outfile] = outblk
        outf.close()
        print("Done writing:", outfile)

    WRITE_BLOCK = write_blk

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
        real part seperated from imag by plus sign;
        also imaginary piece has a 'j'
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
        """write built array to file
        (for use in building disconnected diagrams)
        real part seperated from imag by space
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
                    cnum = complex('{0:.{1}f}'.format(
                        cnum, sys.float_info.dig))
                line = str(tsrc)+" "+str(tdis)+" "+str(cnum.real)+" "+str(
                    cnum.imag)+"\n"
                filen.write(line)
        print("Done writing file:", outfile)
        filen.close()


else:
    print("Error: bad file format specified.")
    print("edit read_file config")
    sys.exit(1)


# util functions, no file interaction
def sum_rows(inmat, avg=False):
    """np.fsum over t_src
    """
    ncol = int(sqrt(inmat.size))
    if avg:
        col = [em.acsum(inmat.item(i, j)
                        for i in range(ncol))/(ncol) for j in range(ncol)]
    else:
        col = [em.acsum(inmat.item(i, j)
                        for i in range(ncol)) for j in range(ncol)]
    return col


def ptostr(ploc):
    """makes momentum array into a string
    """
    return (str(ploc[0])+str(ploc[1])+str(ploc[2])).replace("-", "_")


if __name__ == '__main__':
    pass
