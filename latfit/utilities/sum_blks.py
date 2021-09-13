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
import latfit.utilities.read_file as rf


def get_outblock(coeffs_arr, flag, outfile, time, sent):
    """Get block to write in linear sum of jackknife blocks"""
    outblk = np.array([])
    for pair in coeffs_arr:
        name, coeff = pair
        if flag == sent:
            print("Including:", name, "with coefficient", coeff)
        # do the check after printing out the coefficients
        # so we can check afterwards
        if os.path.isfile(outfile):
            print("Skipping:", outfile)
            print("File exists.")
            continue
        filen = rf.tryblk(name, time)
        if not filen:
            continue
        for i, line in enumerate(filen):
            try:
                line = line.split()
                if len(line) == 1:
                    val = coeff*float(line[0])
                elif len(line) == 2:
                    val = coeff*complex(float(line[0]), float(line[1]))
                else:
                    print("Error: bad block:", filen)
                    break
            except AttributeError:
                val = coeff*line
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
    print("Start of blocks:",
          outdir, "--------------------------------------------")
    onlyfiles = [f for f in
                 listdir('./'+coeffs_arr[0][0])
                 if isfile(join('./'+coeffs_arr[0][0], f))]
    # make new directory if it doesn't exist
    # loop over time slices until there are none left
    sent = object()
    flag = sent
    for time in onlyfiles:
        if re.search('pdf', time):
            print("Skipping 'block' = ", time)
            continue
        outfile = outdir+'/'+time
        outblk = get_outblock(coeffs_arr, flag, outfile, time, sent)
        flag = 0
        if not outblk or os.path.isfile(outfile):
            continue
        rf.write_blk(outblk, outfile, already_checked=True)
    print("End of blocks:", outdir,
          "--------------------------------------------")


def norm_fix(filen):
    """Fix norms due to incorrect coefficients given in production."""
    name = rf.figure(filen)
    norm = 1.0
    if name == 'R':
        norm = 1.0
    elif name == 'T':
        if rf.vecp(filen) and not rf.reverse_p(filen):
            norm = 1.0
        else:
            norm = 1.0
    elif name == 'C':
        norm = 1.0
    elif name == 'D':
        norm = 1.0
    elif name in ('Hbub', 'pioncorr'):
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
    #kaonp = rf.kaonp(filen)
    name = rf.figure(filen)
    #if kaonp:
    #    norm = 0.0
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
        norm = 2.0
    elif name == 'R' and vecp:
        norm = -4.0
    elif name == 'D' and vecp:
        norm = -2.0
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
        norm = 6.0/sqrt(6.0)
    elif name == 'T' and not vecp:
        norm = -5.0/(sqrt(6.0))
    elif name == 'Hbub' and not vecp:
        norm = -1.0
    elif name in ('Bub2', 'bub2'):
        norm = 2.0
    elif name in ('KK2pipi', 'KK2sigma', 'KK2KK'):
        norm = 1.0
    elif name in ('VKK2pipi', 'Vpipi2KK'):
        norm = sqrt(3)
    elif name in ('VKK2sigma', 'Vsigma2KK'):
        norm = -1*sqrt(2)
    elif name == 'VKK2KK':
        norm = 1.0
    elif name == 'DKK2KK':
        norm = 0.0 # dummy coefficient; this "diagram" is only needed for pion (kaon) ratio method
    else:
        norm = None
    return norm


def momtotal(plist, fig=None):
    """Get total center of mass momentum given a list of momenta"""
    if len(plist) == 3 and isinstance(plist[0], (np.integer, int)):
        momret = plist
    elif len(plist) == 2:
        mat = re.search('scalarR', fig)
        nmat = re.search('pol_src', fig)
        if mat or nmat:
            momret = plist[0]
        else:
            momret = plist[1]
    elif len(plist) == 3 and not isinstance(plist[0], (np.integer, int)):
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
KKPIPI = set(['KK2pipi', 'VKK2pipi', 'Vpipi2KK'])
KKSIGMA = set(['KK2sigma', 'VKK2sigma', 'Vsigma2KK'])
KKKK = set(['KK2KK', 'VKK2KK', 'DKK2KK'])
SIGMASIGMA = set(['Hbub', 'Bub2', 'bub2'])
RHORHO = set(['Hbub'])

# key is the name of the output operator
# value: (list of diagrams, list of isospins operator is included in,
# reverse diagram?  True or False)
FILTERLIST = {'pion': (PION, [1], False),
              'pipi': (PIPI, [0, 1, 2], False),
              'pipirho': (PIPIRHO, [1], True),
              'pipisigma': (PIPISIGMA, [0], True),
              'sigmasigma': (SIGMASIGMA, [0], False),
              'rhorho': (RHORHO, [1], False),
              'sigmapipi': (SIGMAPIPI, [0], False),
              'kkpipi': (KKPIPI, [0], False),
              'kksigma': (KKSIGMA, [0], False),
              'kkkk': (KKKK, [0], False),
              'rhopipi': (RHOPIPI, [1], False)}


def get_sep_mom(dlist):
    """Get momentum and time separation lists"""
    momlist = {}
    seplist = {}
    # def is a tuple consisting of list of particles, Isospin,
    # and whether the diagram is a reverse diagram
    for dur in dlist:
        momstr = rf.getmomstr(dur)
        sep = rf.sep(dur)
        if momstr not in momlist:
            momlist[momstr] = set([dur])
        else:
            momlist[momstr].add(dur)
        if sep not in seplist:
            seplist[sep] = set([dur])
        else:
            seplist[sep].add(dur)
        # mom1 = rf.mom(d)
        # if not mom1:
        #    continue
        # if len(mom1) == 2:
        #    momlist.add(tuple(momtotal(mom1, rf.figure(d))))
        # else:
        #    momlist.add(tuple(momtotal(mom1)))
    return seplist, momlist


def get_norm(loop, dur, fixn):
    """Get norm given loop variables, direction dur to check,
    and whether to fix norms (fixn)
    """
    # if momtotal(rf.mom(d), d) != loop.mom:
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
        if fixn:
            norm2 = norm_fix(dur)
        else:
            norm2 = 1.0
        assert norm2 is not None, "norm2 should not be None; bug"
        if norm1 is not None and norm2 is not None:
            norm = norm1*norm2
        else:
            norm = None
    return norm


def get_outdir(loop, dirnum):
    """Get output directory for new linear combination of jackknife blocks.
    """
    if dirnum == 0:
        sepstr = ''
        if loop.sep:
            sepstr = sepstr+"sep"+str(loop.sep)+'/'
        # outdir = loop.opa+"_I"+str(loop.iso)+sepstr+loop.mom
        outdir = 'I'+str(loop.iso)+'/'+sepstr+loop.opa+'_'+loop.mom
    elif dirnum == 1:
        sepstr = '_'
        if loop.sep:
            sepstr = sepstr+"sep"+str(loop.sep)+'_'
        # outdir = loop.opa+"_I"+
        # str(loop.iso)+sepstr+"_momtotal"+rf.ptostr(loop.mom)
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
        if norm is None:
            continue
        coeffs_arr.append((dur, norm))
    return coeffs_arr


def isoproj(fixn, dirnum, dlist=None, stype='ascii'):
    """Isospin projection of jackknife blocks (main)"""
    dur = '.'
    if dlist is None:
        dlist = [os.path.join(dur, o)
                 for o in os.listdir(dur)
                 if os.path.isdir(os.path.join(dur, o))]
    projlist = {}
    seplist, momlist = get_sep_mom(dlist)
    loop = namedtuple('loop', ('opa', 'iso', 'sep', 'mom'))
    for loop.opa in FILTERLIST:
        for loop.iso in FILTERLIST[loop.opa][1]:
            for loop.sep in seplist:
                for loop.mom in momlist:
                    # loop.mom = list(loop.mom)
                    #print(loop.opa, loop.iso, loop.sep, loop.mom)
                    coeffs_arr = get_coeffs_arr(
                        loop, fixn, seplist[loop.sep] & momlist[loop.mom])
                    #print(coeffs_arr)
                    if coeffs_arr == []:
                        continue
                    outdir = get_outdir(loop, dirnum)
                    #print(outdir)
                    if stype == 'ascii':
                        sum_blks(outdir, coeffs_arr)
                    else:
                        projlist[outdir] = coeffs_arr
    if stype == 'ascii':
        print("Done writing jackknife sums.")
    return projlist


if __name__ == '__main__':
    FIXN = input("Need fix norms before summing? True/False?")
    if FIXN == 'True':
        FIXN = True
    elif FIXN == 'False':
        FIXN = False
    else:
        sys.exit(1)
    isoproj(FIXN, 0)
    isoproj(FIXN, 1)
