#!/usr/bin/python3
"""Irrep projection."""
from math import sqrt
import os
import sys
import re
import numpy as np
from latfit.utilities.sum_blks import sum_blks
import latfit.utilities.write_discon as wd
import latfit.utilities.read_file as rf
from latfit.utilities.oplist import OPLIST, AVG_ROWS
from latfit.utilities import exactmean as em


OPLIST_STRIPPED = {}
for OPA in list(OPLIST): # get rid of polarization information
    OPA_STRIP = OPA.split('?')[0]
    OPLIST_STRIPPED[OPA] = OPLIST[OPA]
    if OPA != OPA_STRIP:
        OPLIST_STRIPPED[OPA_STRIP] = OPLIST[OPA]
        del OPLIST_STRIPPED[OPA]
        assert OPA_STRIP in OPLIST_STRIPPED

assert 'T_1_1MINUS_mom000' in OPLIST_STRIPPED, "bug"

def freemomenta(irrep, dim):
    """Get the free momenta for the irrep and dimension"""
    irrep = representative_row(irrep)
    currop = ''
    cdim = -1
    ret = None
    for _, opstr, mom in OPLIST_STRIPPED[irrep]:
        if currop != opstr:
            currop = opstr
            cdim += 1
        if cdim == dim and len(mom) == 2:
            ret = mom
            break
        if cdim == dim:
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
    for irrep in OPLIST_STRIPPED:
        for _, _, mom_comb in OPLIST_STRIPPED[irrep]:
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
    """make mom into a new line separated momentum string """
    return 'int p[0] = '+str(mom[0])+'\nint p[1] = '+\
        str(mom[1])+'\nint p[2] = '+str(mom[2])+'\n'

def free_energies(irrep, pionmass, kaonmass, lbox):
    """return a list of free energies."""
    assert pionmass is not None
    assert kaonmass is not None
    if hasattr(pionmass, '__iter__'):
        assert pionmass[0] is not None
    if hasattr(kaonmass, '__iter__'):
        assert kaonmass[0] is not None
    assert not np.any(np.isnan(pionmass)), str(pionmass)
    assert not np.any(np.isnan(kaonmass)), str(kaonmass)
    retlist = []
    if irrep in AVG_ROWS:
        for irr in AVG_ROWS[irrep]:
            irrep = irr
            break
    opprev = ''
    for _, opa, mom in OPLIST_STRIPPED[irrep]:
        if opa == opprev:
            continue
        if len(mom) != 2:
            continue
        opprev = opa
        energy = 0
        if opa == 'kk':
            toadd = 2*kaonmass
            toadd = np.array(toadd)
            energy += toadd
        else:
            for pin in mom:
                # print(pionmass, pin, lbox)
                if hasattr(pionmass, '__iter__') and np.asarray(pionmass).shape:
                    toadd = [sqrt(i**2+(2*np.pi/lbox)**2*rf.norm2(
                        pin)) for i in pionmass]
                else:
                    toadd = sqrt(pionmass**2+(2*np.pi/lbox)**2*rf.norm2(pin))
                toadd = np.array(toadd)
                energy += toadd
        retlist.append(energy)
    sortedret = []
    for i, mean in enumerate([em.acmean(i) for i in retlist]):
        if em.acmean(retlist[i]) == mean:
            sortedret.append(retlist[i])
    if isinstance(sortedret[0], float):
        sortedret = np.asarray(sorted(sortedret))
    return sortedret

def representative_row(irrep):
    """Get a representative row from an irrep"""
    if irrep in AVG_ROWS:
        for irr in AVG_ROWS[irrep]:
            irrep = irr
            break
    return irrep

def get_comp_str(irrep):
    """Get center of mass momentum of an irrep, return as a string for latfit"""
    takeabs = False
    if rf.mom(irrep) is None:
        takeabs = True
    irrep = representative_row(irrep)
    momtotal = np.array([0, 0, 0])
    assert irrep in OPLIST_STRIPPED, "irrep not found:"+str(irrep)
    for _, _, mom in OPLIST_STRIPPED[irrep]:
        if len(mom) != 2:
            continue
        for pin in mom:
            momtotal += np.array(pin)
        break
    assert np.all(momtotal == rf.mom(irrep)), str(irrep)+" "+str(momtotal)
    # presumably, the average with +/- is taken
    if takeabs:
        momtotal = np.abs(momtotal)
    return 'mom'+rf.ptostr(momtotal)

def firstrep(irrep):
    """Get first representative row"""
    if irrep in AVG_ROWS:
        for irr in AVG_ROWS[irrep]:
            irrep = irr
            break
    return irrep

def mom2ndorder(irrep):
    """Find the two momenta for the second order
    around the world subtraction
    """
    irrep = firstrep(irrep)
    minp = np.inf
    for _, _, mom in OPLIST_STRIPPED[irrep]:
        if isinstance(mom[0], int):
            continue
        p1a, p2a = mom
        minp = min(rf.norm2(p1a), rf.norm2(p2a), minp)
    minp2 = np.inf
    for _, _, mom in OPLIST_STRIPPED[irrep]:
        if isinstance(mom[0], int):
            continue
        p1a, p2a = mom
        if rf.norm2(p1a) == minp or rf.norm2(p2a) == minp:
            continue
        minp2 = min(rf.norm2(p1a), rf.norm2(p2a), minp2)
    ret = None
    mindiff = np.inf
    for _, _, mom in OPLIST_STRIPPED[irrep]:
        if isinstance(mom[0], int):
            continue
        p1a, p2a = mom
        if rf.norm2(p1a) == minp2:
            mindiff = min(mindiff, rf.norm2(p2a)-minp2)
            if mindiff == 0:
                ret = mom
            break
    for _, _, mom in OPLIST_STRIPPED[irrep]:
        if isinstance(mom[0], int):
            continue
        p1a, p2a = mom
        if rf.norm2(p1a) == minp2 and rf.norm2(p2a)-minp2 == mindiff:
            ret = mom if ret is None else ret
    ret = [None, None] if ret is None else ret
    return ret


def generate_checksums(isospin):
    """Generate a sum of expected diagrams for each operator"""
    isospin = int(isospin)
    checks = {}
    for oplist in OPLIST_STRIPPED:
        newl = len(OPLIST_STRIPPED[oplist])
        for _, opa, _ in OPLIST_STRIPPED[oplist]:
            if 'sigma' in opa and isospin != 0:
                newl -= 1
            elif 'kk' in opa and isospin != 0:
                newl -= 1
            elif 'rho' in opa and isospin != 1:
                newl -= 1
        checks[oplist] = newl**2
    return checks


# OPLIST_STRIPPED = {'A0': A0}
PART_LIST = set([])
for opa_out in OPLIST_STRIPPED:
    for item in OPLIST_STRIPPED[opa_out]:
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

def checkdups(opa, src, snk, chkidx, chkidx2):
    """Check for duplicate operators"""
    dup_flag = True
    for pcheck, pcheck2 in zip(src[2], snk[2]):
        if isinstance(pcheck, int):
            dup_flag = pcheck == pcheck2
        elif rf.ptostr(pcheck) != rf.ptostr(pcheck2):
            dup_flag = False
    if dup_flag:
        assert chkidx == chkidx2,\
            "Duplicate operator found in "+str(
                opa)+" "+str(src)+" "+str(snk)


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
                    checkdups(opa, src, snk, chkidx, chkidx2)
                assert cons_mom(src, snk, momchk),\
                    "operator does not conserve momentum "+str(opa)
                part_str = partstr(src[1], snk[1])
                dur = part_str+"_"+momstr(src[2], snk[2])
                dur = re.sub('S_', '', dur)
                dur = re.sub('UU', '', dur)
                for i in range(10):
                    dur = re.sub('U'+str(i), '', dur)
                dur = re.sub('pipipipi', 'pipi', dur)
                dur = sepmod(dur, opa) if stype == 'ascii' else dur
                coeffs_tuple.append((dur, src[0]*snk[0], part_str))
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
    assert len(src) == 3, "bad source/sink spec.:"+str(src)+" "+str(snk)
    assert len(snk) == 3, "bad source/sink spec.:"+str(snk)
    psrc = wd.momtotal(src[2])
    psnk = wd.momtotal(snk[2])
    assert len(psrc) == len(psnk), "bad length for p:"+str(psrc)+" "+str(psnk)
    assert len(psrc) == 3, "bad length for p:"+str(psrc)+" "+str(psnk)
    conssrcsnk = psrc[0] == psnk[0] and psrc[1] == psnk[1] and psrc[2] == psnk[2]
    if momtotal:
        check = momtotal[0] == psnk[0] and momtotal[1] == psnk[1] and momtotal[2] == psnk[2]
    else:
        check = True
    return check and conssrcsnk



def main():
    """Do irrep projection (main)"""
    llist = op_list('hdf5')
    for i in llist:
        print(i)
        #print(l[i])

    print(llist['pipisigma_A_1PLUS_mom000'])
    print(llist['rhorho_T_1_1MINUS?pol=1'])
    #generate_operator_momenta()


if __name__ == "__main__":
    main()
