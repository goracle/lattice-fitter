#!/usr/bin/python3
"""Do isospin projection of jackknife blocks"""
import re
from os import listdir
import os.path
from os.path import isfile, join
from math import sqrt
import sys
from collections import namedtuple
import numpy as np
import read_file as rf

def get_outblock(coeffs_arr, flag, outfile, time, sent):
    """Get block to write in linear sum of jackknife blocks"""
    outblk = np.array([])
    for pair in coeffs_arr:
        name, coeff = pair
        if flag == sent:
            print("Including:", name, "with coefficient", coeff)
        #do the check after printing out the coefficients
        #so we can check afterwards
        if os.path.isfile(outfile):
            print("Skipping:", outfile)
            print("File exists.")
            continue
        try:
            filen = open(name+'/'+time, 'r')
        except IOError:
            print("Error: bad block name in:", name)
            print("block name:", time, "Continuing.")
            continue
        for i, line in enumerate(filen):
            line = line.split()
            if len(line) == 1:
                val = coeff*float(line[0])
            elif len(line) == 2:
                val = coeff*complex(float(line[0]), float(line[1]))
            else:
                print("Error: bad block:", filen)
                break
            try:
                outblk[i] += val
            except IndexError:
                outblk = np.append(outblk, val)
    return outblk

def sum_blks(outdir, coeffs_arr):
    """Given a list of directories of jackknife blocks and coefficients,
    do a linear sum of the blocks line by line and output the resulting
    jackknife blocks to outdir.
    """
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError:
            print("can't create directory:", outdir, "Permission denied.")
            sys.exit(1)
    print("Start of blocks:", outdir, "--------------------------------------------")
    onlyfiles = [f for f in
                 listdir('./'+coeffs_arr[0][0])
                 if isfile(join('./'+coeffs_arr[0][0], f))]
    #make new directory if it doesn't exist
    #loop over time slices until there are none left
    sent = object()
    flag = sent
    for time in onlyfiles:
        if re.search('pdf', time):
            print("Skipping 'block' = ", time)
            continue
        outfile = outdir+'/'+time
        outblk = get_outblock(coeffs_arr, flag, outfile, time, sent)
        flag = 0
        if os.path.isfile(outfile):
            continue
        with open(outfile, 'a') as filen:
            for line in outblk:
                outline = complex('{0:.{1}f}'.format(line, sys.float_info.dig))
                if outline.imag == 0:
                    outline = str(outline.real)+"\n"
                else:
                    outline = str(outline.real)+" "+str(outline.imag)+"\n"
                filen.write(outline)
            print("Done writing:", outfile)
    print("End of blocks:", outdir, "--------------------------------------------")

def norm_fix(filen):
    """Fix norms due to incorrect coefficients given in production."""
    name = rf.figure(filen)
    norm = 1.0
    if name == 'R':
        norm = 1.0
    elif name == 'T':
        if rf.vecp(filen) and not rf.reverse_p(filen):
            norm = -1.0
        else:
            norm = 1.0
    elif name == 'C':
        norm = 1.0
    elif name == 'D':
        norm = 1.0
    elif name == 'Hbub' or name == 'pioncorr':
        norm = 2.0
    elif name == 'scalar-bubble':
        norm = 1.0
    elif name == 'V':
        norm = 4.0
    elif name == 'Cv3':
        norm = 2.0
    elif name == 'Cv3R':
        norm = 2.0
    return norm

def isospin_coeff(filen, iso):
    """Get isospin coefficient"""
    norm = 1.0
    vecp = rf.vecp(filen)
    name = rf.figure(filen)
    if iso == 0:
        norm = iso0(vecp, name)
    elif iso == 1:
        norm = iso1(vecp, name)
    elif iso == 2:
        norm = iso2(vecp, name)
    else:
        print("Error: bad isospin:", iso)
        sys.exit(1)
    return norm

def iso2(vecp, name):
    """Isospin coeff, I = 2"""
    if name == 'D' and not vecp:
        norm = 2.0
    elif name == 'C':
        norm = -2.0
    else:
        norm = None
    return norm

def iso1(vecp, name):
    """Isospin coeff, I = 1"""
    if (name == 'Hbub' and vecp) or name == 'pioncorr':
        norm = 1.0
    elif name == 'T' and vecp:
        norm = 4.0/sqrt(2.0)
    elif name == 'R' and vecp:
        norm = 4.0
    elif name == 'D' and vecp:
        norm = 2.0
    else:
        norm = None
    return norm

def iso0(vecp, name):
    """Isospin coeff, I = 0"""
    if name == 'V':
        norm = 3.0
    elif name == 'D' and not vecp:
        norm = 2.0
    elif name == 'R' and not vecp:
        norm = -6.0
    elif name == 'C':
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
        norm = None
    return norm

def momtotal(plist, fig=None):
    """Get total center of mass momentum given a list of momenta"""
    if len(plist) == 3 and isinstance(plist[0], int):
        momret = plist
    elif len(plist) == 2:
        mat = re.search('scalarR', fig)
        nmat = re.search('pol_src', fig)
        if mat or nmat:
            momret = plist[0]
        else:
            momret = plist[1]
    elif len(plist) == 3 and not isinstance(plist[0], int):
        mom1 = np.array(plist[0])
        mom2 = np.array(plist[1])
        ptotal = list(mom1+mom2)
        momret = ptotal
    else:
        print("Error: bad momentum list:", plist)
        sys.exit(1)
    return momret

PION = set(['pioncorr'])
PIPI = set(['C', 'D', 'R', 'V'])
PIPIRHO = set(['T'])
RHOPIPI = set(['T'])
PIPISIGMA = set(['Cv3R', 'T'])
SIGMAPIPI = set(['Cv3', 'T'])
SIGMASIGMA = set(['Hbub', 'Bub2', 'bub2'])
RHORHO = set(['Hbub'])
FILTERLIST = {'pion':(PION, [1], False),
              'pipi':(PIPI, [0, 1, 2], False),
              'pipirho':(PIPIRHO, [1], True),
              'pipisigma':(PIPISIGMA, [0], True),
              'sigmasigma':(SIGMASIGMA, [0], False),
              'rhorho':(RHORHO, [1], False),
              'sigmapipi':(SIGMAPIPI, [0], False),
              'rhopipi':(RHOPIPI, [1], False)}

def get_sep_mom(dlist):
    """Get momentum and time separation lists"""
    momlist = {}
    seplist = {}
    #def is a tuple consisting of list of particles, Isospin,
    #and whether the diagram is a reverse diagram
    for dur in dlist:
        momstr = rf.getmomstr(dur)
        sep = rf.sep(dur)
        if not momstr in momlist:
            momlist[momstr] = set()
        else:
            momlist[momstr].add(dur)
        if sep not in seplist:
            seplist[sep] = set()
        else:
            seplist[sep].add(dur)
        #mom1 = rf.mom(d)
        #if not mom1:
        #    continue
        #if len(mom1) == 2:
        #    momlist.add(tuple(momtotal(mom1, rf.figure(d))))
        #else:
        #    momlist.add(tuple(momtotal(mom1)))
    return seplist, momlist

def get_norm(loop, dur, fixn):
    """Get norm given loop variables, direction dur to check,
    and whether to fix norms (fixn)
    """
    #if momtotal(rf.mom(d), d) != loop.mom:
    if not rf.figure(dur) in FILTERLIST[loop.opa][0]:
        norm = None
    elif rf.reverse_p(dur) is not FILTERLIST[loop.opa][2]:
        norm = None
    elif re.search('Check', dur) or re.search(
            'Chk', dur) or re.search(
                'chk', dur) or re.search('check', dur):
        norm = None
    else:
        norm1 = isospin_coeff(dur, loop.iso)
        if not norm1:
            norm = None
        elif fixn:
            norm2 = norm_fix(dur)
        else:
            norm2 = 1.0
        norm = norm1*norm2
    return norm

def get_outdir(loop, dirnum):
    """Get output directory for new linear combination of jackknife blocks.
    """
    if dirnum == 0:
        sepstr = ''
        if loop.sep:
            sepstr = sepstr+"sep"+str(loop.sep)+'/'
        #outdir = loop.opa+"_I"+str(loop.iso)+sepstr+loop.mom
        outdir = 'I'+str(loop.iso)+'/'+sepstr+loop.opa+'_'+loop.mom
    elif dirnum == 1:
        sepstr = '_'
        if loop.sep:
            sepstr = sepstr+"sep"+str(loop.sep)+'_'
        #outdir = loop.opa+"_I"+str(loop.iso)+sepstr+"_momtotal"+rf.ptostr(loop.mom)
        outdir = loop.opa+"_I"+str(loop.iso)+sepstr+loop.mom
    else:
        print("Error: bad flag specified. dirnum =", dirnum)
        sys.exit(1)
    return outdir

def get_coeffs_arr(loop, fixn, dlist):
    """Get array of coefficients for jackknife block sum.
    """
    coeffs_arr = []
    for dur in dlist:
        norm = get_norm(loop, dur, fixn)
        if not norm:
            continue
        coeffs_arr.append((dur, norm))
    return coeffs_arr

def main(fixn, dirnum):
    """Isospin projection of jackknife blocks (main)"""
    dur = '.'
    dlist = [os.path.join(dur, o) for o in os.listdir(dur) if os.path.isdir(os.path.join(dur, o))]
    seplist, momlist = get_sep_mom(dlist)
    loop = namedtuple('loop', ('opa', 'iso', 'sep', 'mom'))
    for loop.opa in FILTERLIST:
        for loop.iso in FILTERLIST[loop.opa][1]:
            for loop.sep in seplist:
                for loop.mom in momlist:
                    #loop.mom = list(loop.mom)
                    coeffs_arr = get_coeffs_arr(
                        loop, fixn, seplist[loop.sep] & momlist[loop.mom])
                    if coeffs_arr == []:
                        continue
                    outdir = get_outdir(loop, dirnum)
                    sum_blks(outdir, coeffs_arr)
    print("Done writing jackknife sums.")

if __name__ == '__main__':
    FIXN = input("Need fix norms before summing? True/False?")
    if FIXN == 'True':
        FIXN = True
    elif FIXN == 'False':
        FIXN = False
    else:
        sys.exit(1)
    main(FIXN, 0)
    main(FIXN, 1)
