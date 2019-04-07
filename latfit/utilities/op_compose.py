#!/usr/bin/python3
"""Irrep projection."""
from math import sqrt
import os
import sys
import re
import numpy as np
from sum_blks import sum_blks
import write_discon as wd
import read_file as rf
from oplist import *

for opa in list(OPLIST): # get rid of polarization information
    opa_strip = opa.split('?')[0]
    if opa != opa_strip:
        OPLIST[opa_strip] = OPLIST[opa] 
        del OPLIST[opa]


def freemomenta(irrep, dim):
    """Get the free momenta for the irrep and dimension"""
    irrep = representative_row(irrep)
    currop = ''
    cdim = -1
    for _, opstr, mom in OPLIST[irrep]:
        if currop != opstr:
            currop = opstr
            cdim += 1
        if cdim == dim and len(mom) == 2:
            ret = mom
            break
        elif cdim == dim:
            dim += 1
    return ret

def momstr(psrc, psnk):
    """Take psrc and psnk and return a diagram string of the combination.
    """
    pipi = ''
    if len(psrc) == 2 and len(psnk) == 2:
        vertices = 4
    elif len(psrc) == 3 and isinstance(
            psrc[0], (np.integer, int)) and len(psnk) == 2:
        vertices = 3
        pipi = 'snk'
    elif len(psnk) == 3 and isinstance(
            psnk[0], (np.integer, int)) and len(psrc) == 2:
        vertices = 3
        pipi = 'src'
    elif len(psnk) == len(psrc) and len(psrc) == 3 and isinstance(
            psrc[0], (np.integer, int)):
        vertices = 2
    else:
        pstr = None
    if vertices == 4:
        pstr = 'mom1src'+rf.ptostr(
            #psrc[0])+'_mom2src'+rf.ptostr(
            #    psrc[1])+'_mom1snk'+rf.ptostr(psnk[0])
            psrc[1])+'_mom2src'+rf.ptostr(
                psrc[0])+'_mom1snk'+rf.ptostr(psnk[1])
    elif vertices == 3 and pipi == 'src':
        #pstr = 'momsrc'+rf.ptostr(psrc[0])+'_momsnk'+rf.ptostr(psnk)
        pstr = 'momsrc'+rf.ptostr(psrc[1])+'_momsnk'+rf.ptostr(psnk)
    elif vertices == 3 and pipi == 'snk':
        #pstr = 'momsrc'+rf.ptostr(psrc)+'_momsnk'+rf.ptostr(psnk[0])
        pstr = 'momsrc'+rf.ptostr(psrc)+'_momsnk'+rf.ptostr(psnk[1])
    elif vertices == 2:
        if psrc[0] != psnk[0] or psrc[1] != psnk[1] or psrc[2] != psnk[2]:
            pstr = None
        else:
            pstr = 'mom'+rf.ptostr(psrc)
    return pstr

def write_mom_comb():
    """Write the momentum combination vml
    (list of allowed two particle momenta)"""
    twoplist = {}
    for irrep in OPLIST:
        for _, _, mom_comb in OPLIST[irrep]:
            if len(mom_comb) == 2: # skip the sigma
                twoplist[str(mom_comb)] = mom_comb
    begin = 'Array moms[2] = {\nArray moms[0] = {\nArray p[3] = {\n'
    middle = '}\n}\nArray moms[1] = {\nArray p[3] = {\n'
    end = '}\n'*4
    with open('mom_comb.vml', 'w') as fn1:
        fn1.write('class allowedCombP mom_comb = {\n')
        fn1.write('Array momcomb['+str(len(twoplist))+'] = {\n')
        for i, comb in enumerate(sorted(twoplist)):
            fn1.write('Array momcomb['+str(i)+'] = {\n')
            fn1.write(begin)
            fn1.write(ptonewlinelist(twoplist[comb][0]))
            fn1.write(middle)
            fn1.write(ptonewlinelist(twoplist[comb][1]))
            fn1.write(end)
        fn1.write('}\n}')

def ptonewlinelist(mom):
   """make mom into a new line separated momentum string
   """ 
   return 'int p[0] = '+str(mom[0])+'\nint p[1] = '+\
       str(mom[1])+'\nint p[2] = '+str(mom[2])+'\n'

def free_energies(irrep, pionmass, lbox):
    """return a list of free energies."""
    retlist = []
    if irrep in AVG_ROWS:
        for irr in AVG_ROWS[irrep]:
            irrep = irr
            break
    opprev = ''
    for _, opa, mom in OPLIST[irrep]:
        if opa == opprev:
            continue
        if len(mom) != 2:
            continue
        opprev = opa
        energy = 0
        for pin in mom:
            # print(pionmass, pin, lbox)
            energy += sqrt(pionmass**2+(2*np.pi/lbox)**2*rf.norm2(pin))
        retlist.append(energy)
    return sorted(retlist)

def representative_row(irrep):
    if irrep in AVG_ROWS:
        for irr in AVG_ROWS[irrep]:
            irrep = irr
            break
    return irrep

def get_comp_str(irrep):
    """Get center of mass momentum of an irrep, return as a string for latfit"""
    retlist = []
    irrep = representative_row(irrep)
    opprev = ''
    momtotal = np.array([0, 0, 0])
    for _, _, mom in OPLIST[irrep]:
        if len(mom) != 2:
            continue
        for pin in mom:
            momtotal += np.array(pin)
        break
    return 'mom'+rf.ptostr(momtotal)

def mom2ndorder(irrep):
    """Find the two momenta for the second order
    around the world subtraction
    """
    retlist = []
    if irrep in AVG_ROWS:
        for irr in AVG_ROWS[irrep]:
            irrep = irr
            break
    opprev = ''
    momtotal = np.array([0, 0, 0])
    minp = np.inf
    for _, _, mom in OPLIST[irrep]:
        if isinstance(mom[0], int):
            continue
        p1, p2 = mom
        minp = min(rf.norm2(p1), rf.norm2(p2), minp)
    minp2 = np.inf
    for _, _, mom in OPLIST[irrep]:
        if isinstance(mom[0], int):
            continue
        p1, p2 = mom
        if rf.norm2(p1) == minp or rf.norm2(p2) == minp:
            continue
        minp2 = min(rf.norm2(p1), rf.norm2(p2), minp2)
    ret = None
    mindiff = np.inf
    for _, _, mom in OPLIST[irrep]:
        if isinstance(mom[0], int):
            continue
        p1, p2 = mom
        if rf.norm2(p1) == minp2:
            mindiff = min(mindiff, rf.norm2(p2)-minp2)
            if mindiff == 0:
                ret = mom
            break
    for _, _, mom in OPLIST[irrep]:
        if isinstance(mom[0], int):
            continue
        p1, p2 = mom
        if rf.norm2(p1) == minp2 and rf.norm2(p2)-minp2 == mindiff:
            ret = mom if ret is None else ret
    return ret


def generateChecksums(isospin):
    """Generate a sum of expected diagrams for each operator"""
    isospin = int(isospin)
    checks = {}
    for oplist in OPLIST:
        newl = len(OPLIST[oplist])
        for coeff, opa, mom in OPLIST[oplist]:
            if 'sigma' in opa and isospin != 0:
                newl -= 1
            elif 'rho' in opa and isospin != 1:
                newl -= 1
        checks[oplist] = newl**2
    return checks


# OPLIST = {'A0': A0}
PART_LIST = set([])
for opa_out in OPLIST:
    for item in OPLIST[opa_out]:
        PART_LIST.add(item[1])


def partstr(srcpart, snkpart):
    """Get string from particle strings at source and sink"""
    if srcpart == snkpart and srcpart == 'pipi':
        particles = 'pipi'
    else:
        particles = snkpart+srcpart
    return particles


PART_COMBS = set([])
for srcout in PART_LIST:
    for snkout in PART_LIST:
        PART_COMBS.add(partstr(srcout, snkout))


def sepmod(dur, opa):
    """make different directory name for different time separations
    (probably what this does.)
    """
    if not os.path.isdir(dur):
        if not os.path.isdir('sep4/'+dur):
            print("For op:", opa)
            print("dir", dur, "is missing")
            sys.exit(1)
    else:
        dur = 'sep4/'+dur
    return dur


def op_list(stype='ascii'):
    """Compose irrep operators at source and sink to do irrep projection.
    """
    projlist = {}
    for opa in OPLIST:
        coeffs_tuple = []
        momchk = rf.mom(opa) if 'mom' in opa else None
        for chkidx, src in enumerate(OPLIST[opa]):
            for chkidx2, snk in enumerate(OPLIST[opa]):
                if src[1] == snk[1]:
                    dup_flag = True
                    for pcheck, pcheck2 in zip(src[2], snk[2]):
                        if isinstance(pcheck, int):
                            dup_flag = pcheck == pcheck2
                        elif rf.ptostr(pcheck) != rf.ptostr(pcheck2):
                            dup_flag = False
                    if dup_flag:
                        assert chkidx == chkidx2, "Duplicate operator found in "+str(opa)+" "+str(src)+" "+str(snk)
                assert cons_mom(src, snk, momchk), "operator does not conserve momentum "+str(opa)
                part_str = partstr(src[1], snk[1])
                coeff = src[0]*snk[0]
                p_str = momstr(src[2], snk[2])
                dur = part_str+"_"+p_str
                dur = re.sub('S_', '', dur)
                dur = re.sub('UU', '', dur)
                for i in range(10):
                    dur = re.sub('U'+str(i), '', dur)
                dur = re.sub('pipipipi', 'pipi', dur)
                if stype == 'ascii':
                    dur = sepmod(dur, opa)
                coeffs_tuple.append((dur, coeff, part_str))
        coeffs_arr = []
        if stype == 'ascii':
            print("trying", opa)
        for parts in PART_COMBS:
            outdir = parts+"_"+opa
            coeffs_arr = [(tup[0], tup[1])
                          for tup in coeffs_tuple if tup[2] == parts]
            if not coeffs_arr:
                continue
            if stype == 'ascii':
                print("Doing", opa, "for particles", parts)
                sum_blks(outdir, coeffs_arr)
            else:
                projlist[outdir] = coeffs_arr
    if stype == 'ascii':
        print("End of operator list.")
    return projlist

def cons_mom(src, snk, momtotal=None):
    """Check for momentum conservation"""
    psrc = wd.momtotal(src[2])
    psnk = wd.momtotal(snk[2])
    conssrcsnk = psrc[0] == psnk[0] and psrc[1] == psnk[1] and psrc[2] == psnk[2]
    if momtotal:
        check = momtotal[0] == psnk[0] and momtotal[1] == psnk[1] and momtotal[2] == psnk[2]
    else:
        check = True
    return check and conssrcsnk
    


def main():
    """Do irrep projection (main)"""
    l = op_list('hdf5')
    for i in l:
        print(i)
        #print(l[i])

    print(l['pipisigma_A_1PLUS_mom000'])
    print(l['rhorho_T_1_1MINUS?pol=1'])
    generateOperatorMomenta()


if __name__ == "__main__":
    main()
