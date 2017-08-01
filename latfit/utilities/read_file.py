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
import h5py

FORMAT = 'ASCII'

######regex on filename stuff
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

def basename(filen):
    """get basename of file
    """
    mat = re.search('traj_[B0-9]+_(.*)', filen)
    if not mat:
        return None
    return mat.group(1)

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
        print("Error: bad filename= '"+str(
            filename)+"' no momenta found.  Attempting to continue.")
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


#################file io stuff

if FORMAT == 'ASCII':
    #####test functions
    def discon_test(filename):
        filen = open(filename, 'r')
        for line in filen:
            if len(line.split()) < 4:
                print("Disconnected.  Skipping.")
                return False
        return True

    def tryblk(name, time):
        try:
            filen = open(name+'/'+time, 'r')
        except IOError:
            print("Error: bad block name in:", name)
            print("block name:", time, "Continuing.")
            return False
        return filen

    ####read file
    def getaux_filestrs(filename):
        """gets the array from the file, but keeps the values as strings
        """
        len_t = find_dim(filename)
        tsep = pion_sep(filename)
        nmomaux = nmom(filename)
        out = np.zeros((len_t, len_t), dtype=np.object)
        filen = open(filename, 'r')
        for line in filen:
            lsp = line.split(' ')#lsp[0] = tsrc, lsp[1] = tdis
            tsrc = int(lsp[0])
            tdis = int(lsp[1])
            if nmomaux == 3:
                tsrc2 = (tsrc+tdis+tsep)%len_t
                tdis2 = (3*len_t-2*tsep-tdis)%len_t
            elif nmomaux == 2:
                tsrc2 = (tdis+tsrc)%len_t
                tdis2 = (2*len_t-tsep-tdis)%len_t
            elif nmomaux == 1:
                tsrc2 = (tdis+tsrc)%len_t
                tdis2 = (len_t-tdis)%len_t
            else:
                print("Error: bad filename, error in getaux_filestrs")
                sys.exit(1)
            out[tsrc2][tdis2] = str(lsp[2])+" "+str(lsp[3]).rstrip()
        return out

    def get_block_data(filen, onlydirs):
        """Get array of jackknife block data (from single time slice file, e.g.)
        """
        retarr = np.array([], dtype=complex)
        filen = open(filen, 'r')
        for line in filen:
            lsp = line.split()
            if len(lsp) == 2:
                retarr = np.append(retarr, complex(float(lsp[0]), float(lsp[1])))
            elif len(lsp) == 1:
                retarr = np.append(retarr, complex(lsp[0]))
            else:
                print("Not a jackknife block.  exiting.")
                print("cause:", filen)
                print(onlydirs)
                sys.exit(1)
        return retarr

    def get_linejk(traj, basename2, time, trajl):
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
        #back = front
        filen = open(filename, 'r')
        for line in filen:
            lsp = line.split()
            if len(lsp) != 4:
                return None
            #lsp[0] = tsrc, lsp[1] = tdis
            tsrc = int(lsp[0])
            #tsnk = (int(lsp[0])+int(lsp[1]))%len_t
            tdis = int(lsp[1])
            #tdis2 = len_t-int(lsp[1])-1
            front.itemset(tsrc, tdis, complex(float(lsp[2]), float(lsp[3])))
            #back.itemset(tsnk, tdis2, complex(float(lsp[2]), float(lsp[3])))
        if sum_tsrc:
            #return sum_rows(front), sum_rows(back)
            retarr = sum_rows(front, True)
        else:
            retarr = front
        return retarr
            #return front, back

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

    ##test functions which read

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

    def numlines(fn):
        """Find number of lines in a file"""
        return sum(1 for line in open(fn))

    #####write file

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
                    line = complex('{0:.{1}f}'.format(line, sys.float_info.dig))
                    line = str(line.real)+" "+str(line.imag)+"\n"
                filen.write(line)
            print("Done writing:", outfile)

    write_block = write_blk

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
                    cnum = complex('{0:.{1}f}'.format(cnum, sys.float_info.dig))
                line = str(tsrc)+" "+str(tdis)+" "+str(cnum.real)+" "+str(
                    cnum.imag)+"\n"
                filen.write(line)
        print("Done writing file:", outfile)
        filen.close()

elif FORMAT == 'HDF5':
    #####test functions
    def discon_test(filename):
        filenp = h5py.File(filename, 'r')
        filen = h5py[filename]
        dimf = len(filen)
        if filen.shape != (dimf, dimf):
            print("Disconnected.  Skipping.")
            return False
        return True
    #TODO
    def tryblk(name, time):
        try:
            filen = open(name+'/'+time, 'r')
        except IOError:
            print("Error: bad block name in:", name)
            print("block name:", time, "Continuing.")
            return False
        return filen
    #####write file

    def write_blk(outblk, outfile, already_checked=False):
        """Write numerical array of jackknife block to file"""
        if not already_checked:
            if os.path.isfile(outfile):
                print("Skipping:", outfile)
                print("File exists.")
                return
        outf = h5py.File(outfile, 'w')
        outf[outfile]=outblk
        outf.close()
        print("Done writing:", outfile)

    write_block = write_blk

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
                    cnum = complex('{0:.{1}f}'.format(cnum, sys.float_info.dig))
                line = str(tsrc)+" "+str(tdis)+" "+str(cnum.real)+" "+str(
                    cnum.imag)+"\n"
                filen.write(line)
        print("Done writing file:", outfile)
        filen.close()

else:
    print("Error: bad file format specified.")
    print("edit read_file config")
    sys.exit(1)


#####util functions, no file interaction
def sum_rows(inmat, avg=False):
    """np.fsum over t_src
    """
    ncol = int(sqrt(inmat.size))
    if avg:
        col = [np.sum(inmat.item(i, j) for i in range(ncol))/(ncol) for j in range(ncol)]
    else:
        col = [np.sum(inmat.item(i, j) for i in range(ncol)) for j in range(ncol)]
    return col

def ptostr(ploc):
    """makes momentum array into a string
    """
    return (str(ploc[0])+str(ploc[1])+str(ploc[2])).replace("-", "_")


if __name__ == '__main__':
    pass
