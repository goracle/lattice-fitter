#!/usr/bin/python3

#to get files
import sys
#to process file list
from os import listdir
import os.path
from os.path import isfile, join
import numpy as np
from math import sqrt
import re
from combine import comb_dis
import warnings

#get polarizations from file
def pol(filename):
    m = re.search('pol_snk_(\d)', filename)
    n = re.search('pol_src-snk_(\d)-(\d)',filename)
    if m:
        return int(m.group(1))
    elif n:
        return int(n.group(1)), int(n.group(2))
    else:
        return None

#get trajectory from file
def traj(filename):
    m = re.search('traj_([B0-9]+)_[A-Za-z]{1}',filename)
    if m:
        return m.group(1)
    else:
        warnings.warn("Warning: filename:'"+filename+"', has no configuration info")
        return None

#determine if we are dealing with a reverse diagram
def reverseP(filename):
    if re.search('scalarR',filename):
        return True
    elif figure(filename) == 'T' and re.search('pol_src',filename):
        return True
    elif figure(filename) == 'Cv3R':
        return True
    else:
        return False
    
#determine if file contains check data on vector diagram
def checkp(filename):
    m = re.search('_vecCheck_',filename)
    if m:
        return True
    else:
        return False

#is the diagram a vector
def vecp(filename):
    m = re.search('_vec_',filename)
    if m:
        return True
    else:
        return False

#return figure name
def figure(filename):
    h=re.search('([A-Za-z0-9]*)corr',filename)
    l=re.search('Figure(.*?)$',filename)
    k=re.search('Figure_(.*?)$',filename)
    m = re.search('Figure(.*?)_',filename)
    n = re.search('Figure_(.*?)_',filename)
    if n:#don't reverse m and n order, helps for some reason
        return n.group(1)
    elif m:
        return m.group(1)
    elif l:
        return l.group(1)
    elif k:
        return k.group(1)
    #just a two point single particle correlator
    elif h:
        return h.group(0)
    else:
        warnings.warn("Warning: bad filename, no figure name: "+filename)
        return None

#how many momenta returned?
def nmom_arr(pret):
    n=len(pret)
    if type(pret[0]) is int and n==3:
        return 1
    elif type(pret[0]) is not int:
        return n
    else:
        print("Error: bad momentum container.")
        exit(1)

def nmom(filename):
    return nmom_arr(mom(filename))

#help function for mom(filename).  gets int momentum array from regex match
def procmom(mstr):
    b = re.findall('_*\d', mstr)
    if (len(b) != 3):
            print("Error: bad filename, number of momenta")
            exit(1)
    p = [ int(b[i].replace('_','-')) for i in range(len(b))]
    return p

def getmomstr(filename):
    m = re.search('mom.*',filename)
    if m:
        return m.group(0)
    else:
        warnings.warn("Error: no momentum in filename: "+filename)
        return None

#returns momentum of filename in the form of an int array
def mom(filename):
    k = re.search('mom1((_?\d+){3})',filename)
    l = re.search('momsrc((_?\d+){3})',filename)
    m = re.search('mom1src((_?\d+){3})',filename)
    n = re.search('mom((_?\d+){3})',filename)
    if k:
        p1 = procmom(k.group(1))
        k = re.search('mom2((_?\d+){3})',filename)
        if k:
            p2 = procmom(k.group(1))
            return p1,p2
        else:
            print("Error: bad filename, p2 not found",filename)
            exit(1)
    elif l:
        psrc = procmom(l.group(1))
        l = re.search('momsnk((_?\d+){3})',filename)
        if l:
            psnk = procmom(l.group(1))
            return psrc,psnk
        else:
            print("Error: bad filename, psnk not found",filename)
            exit(1)
    elif m:
        psrc1 = procmom(m.group(1))
        m = re.search('mom2src((_?\d+){3})',filename)
        if m:
            psrc2 = procmom(m.group(1))
            m = re.search('mom1snk((_?\d+){3})',filename)
            if m:
                psnk1 = procmom(m.group(1))
                return psrc1,psrc2,psnk1
            else:
                print("Error: bad filename, psnk1 not found",filename)
                exit(1)
        else:
            print("Error: bad filename, psrc2 not found.",filename)
            exit(1)
    elif n:
        p = procmom(n.group(1))
        return p
    else:
        print("Error: bad filename= '"+str(filename)+"' no momenta found.  Attempting to continue.")
        return None
        #exit(1)
    
#returns pion separation (not to be confused with distance)
#i.e. tdis is the separation in time between src/snk
#pion separation is how far apart the pions are in time but mostly localized to one time slice
def pion_sep(filename):
    m = re.search('_sep(\d+)_',filename)
    if m:
        return int(m.group(1))
    else:
        return None

def sep(filename):
    return pion_sep(filename)

#np.fsum over t_src
def sum_rows(inmat, avg=False):
    ncol=int(sqrt(inmat.size))
    if avg:
        col = [np.sum(inmat.item(i,j) for i in range(ncol))/(ncol) for j in range(ncol)]
    else:
        col = [np.sum(inmat.item(i,j) for i in range(ncol)) for j in range(ncol)]
    return col

#get dimensions of the matrix from the sqrt(num_rows) of the file
def find_dim(filename):
    nl = sum(1 for line in open(filename))
    if(sqrt(nl) == int(sqrt(nl))):
        Lt=int(sqrt(nl))
        return Lt
    else:
        print("Error: Non-square matrix, nl=", nl)
        return None

#write vector of strings
def write_vec_str(data,outfile):
    if(os.path.isfile(outfile)):
        print("Skipping:", outfile)
        print("File exists.")
        return
    fn = open(outfile, 'w')
    Lt = len(data)
    for tdis in range(Lt):
        c=data[tdis]
        if not type(c) is str:
            c='{0:.{1}f}'.format(c,sys.float_info.dig)
        line = str(tdis)+" "+str(c)+"\n"
        fn.write(line)
    print("Done writing file:", outfile)
    fn.close()

#write matrix of strings
def write_mat_str(data,outfile):
    if(os.path.isfile(outfile)):
        print("Skipping:", outfile)
        print("File exists.")
        return
    fn = open(outfile, 'w')
    Lt = len(data[0])
    if Lt != len(data):
        print("Error: non-square matrix.")
        exit(1)
    for tsrc in range(Lt):
        for tdis in range(Lt):
            c=data[tsrc][tdis]
            if not type(c) is str:
                c='{0:.{1}f}'.format(c,sys.float_info.dig)
            line = str(tsrc)+" "+str(tdis)+" "+str(c)+"\n"
            fn.write(line)
    print("Done writing file:", outfile)
    fn.close()

#scan the output directory where all the files to be processed into the GEVP array are stored
#returns list of trajectories and momenta in run
def scan_dir():
    trajl = []
    plist = []
    onlyfiles=[f for f in listdir('.') if isfile(join('.',f))]
    for fn in onlyfiles:
        trajl.append(traj(fn))
        trajl=list(set(trajl))
        momret=mom(fn)
        for pnew in momret:
            if type(pnew) is int:
                plist.append(momret)
            elif len(pnew)==3 and type(momret[0]) is int:
                plist.append(pnew)
            else: #shouldn't get here
                print("Error: bad filename, momentum somehow in bad format.")
        plist = np.vstack({tuple(row) for row in plist})
        trajl = list(set(trajl))
    return trajl, plist

#makes momentum array into a string
def ptostr(p):
    return (str(p[0])+str(p[1])+str(p[2])).replace("-","_")

#get filename based on search string.  probably a library function does this already
def find_target(target):
    onlyfiles=[f for f in listdir('.') if isfile(join('.',f))]
    for fn in onlyfiles:
        if fn == target:
            return target
    return None

#return scalar bubble filename
def find_scalar_bub(conf,psnk):
    target = "traj_"+str(conf)+"_Figure_scalar-bubble_mom"+ptostr(psnk)
    return find_target(target)

#return V diagram filename
def find_V(conf,p1,p2,sep=1):
    target = "traj_"+str(conf)+"_Figure_Vdis_sep"+str(sep)+"_mom1"+ptostr(p1)+"_mom2"+ptostr(p2)
    return find_target(target)

#write built array to file (for use in building disconnected diagrams)
def write_arr(data,outfile):
    if(os.path.isfile(outfile)):
        print("Skipping:", outfile)
        print("File exists.")
        return
    fn = open(outfile, 'w')
    Lt = len(data[0])
    if Lt != len(data):
        print("Error: non-square matrix. outfile=", outfile)
        exit(1)
    for tsrc in range(Lt):
        for tdis in range(Lt):
            c=data[tsrc][tdis]
            if not type(c) is str:
                c=complex('{0:.{1}f}'.format(c,sys.float_info.dig))
            line = str(tsrc)+" "+str(tdis)+" "+str(c.real)+" "+str(c.imag)+"\n"
            fn.write(line)
    print("Done writing file:", outfile)
    fn.close()


#replace momenta in filename
def pchange(filename,pnew):
    pold=mom(filename)
    nold=nmom_arr(pold)
    nnew=nmom_arr(pnew)
    t=traj(filename)
    fn=re.sub("traj_.*?_","traj_TEMPsafe_",filename)
    if nnew!=nold:
        print("Error: filename momentum mismatch")
        exit(1)
    n=nold
    if n==1:
        fn=fn.replace(ptostr(pold),ptostr(pnew))
    elif n==2 or n==3:
        for i in range(n):
            fn=fn.replace(ptostr(pold[i]),"temp"+str(i),1)
        for i in range(n):
            fn=fn.replace("temp"+str(i),ptostr(pnew[i]),1)
    else:
        print("Error: bad filename for momentum replacement specified.")
        exit(1)
    fn=re.sub("TEMPsafe",str(t),fn)
    return fn

#helper function; find remaining momentum
def remp(p1,p2,p3=[0,0,0]):
    return np.array(p1)+np.array(p2)-np.array(p3)


##todo below this line

def compute_coeff(fn, I=0):
    fig, figb, name, vp = proc_and_fix_norm(fn)
    coeff = None
    coeffb = None
    g = numpy.zeros((3,3))
    if(name == 'R'):
        if(vp):
            coeff = 2.0
        elif(I == 0):
            coeff = -3.0
            g[0][0] -= 3.0 * fig
    elif(name == 'T'):
        if(vp):
            coeff = 2.0/sqrt(2.0)
            coeffb = -1.0*coeff
        elif(I == 0):
            coeff = -5.0/sqrt(6.0)
    elif(name == 'C'):
        if(I==1):
            coeff = -2.0
        elif(I==0):
            g[0][0] += fig
            coeff = 1.0
    elif(name == 'D'):
        coeff = 1.0
    elif(name == 'Hbub'):
        if(I == 0):
            coeff = -1.0
        elif(I == 1):
            coeff = 1.0
    elif(name == 'bub2' and I==0):
        g[1][1] += 2.0 * fig
        coeff = 2.0
    elif(name == 'V' and I==0):
        coeff = 3.0
    elif(name == 'Cv3' and I==0):
        coeff = 5.0/sqrt(6.0)
    if coeff != None and coeff == None:
        coeffb = coeff
    fig *= coeff
    figb *= coeffb
    #return gevp elements, not this, TODO!
    return fig, figb
        
def main():
    print("untested, exiting.")
    exit(0)
    args=len(sys.argv)
    ar=sys.argv
    print("Processing files...")
    trajl, plist = scan_dir()
    create_dis_figs(trajl,plist)
    for conf in trajl:
        for psrc in plist:
            count = 0
            for psnk in plist:
                count += 1
                for fn in onlyfiles:
                    if traj(fn) != conf:
                        continue
                    momret = mom(fn)
                    if len(momret) == 2:
                        p1,p2 = momret
                        if p1 != psrc or p2 != psnk:
                            continue
                    elif len(momret) == 3 and (psrc != momret or count > 1):
                        #this is for files which inherently only have one momentum specified
                        #e.g. one particle -> one particle
                        continue
                    else:
                        print("Skipping.  No momenta found.")
                        continue
                    #I=0
                    fig, figb = compute_coeff(fn, 0)
                    cpp += fig
                    #I=2
                    fig, figb = compute_coeff(fn, 2)
                    csp += fig
                    cps += figb
                    #I=1 (when we figure it out)
                    fig, figb = compute_coeff(fn, 1)

if __name__ == '__main__':
    main()
