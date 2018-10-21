#!/usr/bin/python3
"""Write jackknife blocks from h5py files"""
import sys
import time
import os
import glob
import re
import numpy as np
from mpi4py import MPI
import itertools
import h5py
import read_file as rf
from sum_blks import isoproj
import op_compose as opc
from op_compose import AVG_ROWS
import write_discon as wd
import aux_write as aux
import math
import avg_hdf5
import glob

MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

# run a test on a 4^4 latice
TEST44 = True
TEST44 = False

# run a test on a 24c x 64 lattice
TEST24C = True
TEST24C = False
TEST44 = True if TEST24C else TEST44

# exclude all diagrams derived from aux symmetry
NOAUX = False
NOAUX = True
# aux testing, overwrite the production set with aux diagrams
# also, don't exit on finding aux pairs in the base dataset
AUX_TESTING = True
AUX_TESTING = False

# representative hdf5 file, to get info about lattice
PREFIX = 'traj_'
EXTENSION = 'hdf5'
FNDEF = PREFIX+'250.'+EXTENSION
GNDEF = PREFIX+'250.'+EXTENSION
HNDEF = PREFIX+'250.'+EXTENSION
FNDEF = PREFIX+'1090.'+EXTENSION
GNDEF = PREFIX+'1110.'+EXTENSION
HNDEF = PREFIX+'1230.'+EXTENSION
if TEST44:
    FNDEF = PREFIX+'4540.'+EXTENSION
    GNDEF = PREFIX+'4540.'+EXTENSION
    HNDEF = PREFIX+'4540.'+EXTENSION
if TEST24C:
    FNDEF = PREFIX+'2460.'+EXTENSION
    GNDEF = PREFIX+'2460.'+EXTENSION
    HNDEF = PREFIX+'2460.'+EXTENSION
# size of lattice in time, lattice units
LT = 64 if not TEST44 else 4
LT = LT if not TEST24C else 64
TSEP = 3 if not TEST44 else 1
TSEP = TSEP if not TEST24C else 3
# format for files; don't change
STYPE = 'hdf5'
# precomputed indexing matrices; DON'T CHANGE
ROWST = np.tile(np.arange(LT), (LT, 1))
ROWS = np.tile(np.arange(LT), (LT, 1)).T
COLS = np.array([np.roll(np.arange(LT), -i, axis=0) for i in range(LT)])

# options concerning how bubble subtraction is done
TAKEREAL = False  # take real of bubble if momtotal=0
STILLSUB = False  # don't do subtraction on bubbles with net momentum
TIMEAVGD = False  # do a time translation average (bubble is scalar now)
NOSUB = False  # don't do any subtraction if true; set false if doing GEVP

# other config options
THERMNUM = 0  # eliminate configs below this number to thermalize
THERMNUM = THERMNUM if not TEST44 else 4539  # eliminate configs below this number to thermalize
THERMNUM = THERMNUM if not TEST24C else 0
#assert not THERMNUM, "thermnum not equal to 0="+str(THERMNUM)
TSTEP = 8 if not TEST44 else 1  # we only measure every TSTEP time slices to save on time
TSTEP = TSTEP if not TEST24C else 8
# max time distance between (inner) particles
TDIS_MAX = 64
TDIS_MAX = 16


# DO NOT CHANGE IF NOT DEBUGGING
# do subtraction in the old way
OUTERSUB = False  # (True): <AB>-<A><B>.  New way (False): <A-<A>><B-<B>>
JACKBUB = True  # keep true for correctness; false to debug incorrect results
assert(not(OUTERSUB and JACKBUB)), "Not supported!  new:JACKBUB = True, " + \
    "OUTERSUB = False, " + " debug:JACKBUB = False, OUTERSUB = False"
# FOLD = True # average about the mirror point in time (True)
FOLD = True
# Print isospin and irrep projection coefficients of operator to be written
PRINT_COEFFS = True
CONJBUB = False
# ANTIPERIODIC in time (false for mesons,
# the true option only happens for tanh shapes of the correlator);
# this is a bad feature, keep it false!
ANTIPERIODIC = False

# Filter out the cross momenta
# (back to back x momenta going to back to back y momenta, e.g.)
FILTER_OUT_CROSS_MOMENTA = True
FILTER_OUT_CROSS_MOMENTA = False

# diagram to look at for bubble subtraction test
# TESTKEY = 'FigureV_sep4_mom1src001_mom2src010_mom1snk010'
# TESTKEY = 'FigureV_sep4_mom1src000_mom2src001_mom1snk000'
# TESTKEY = 'FigureV_sep4_mom1src001_mom2src000_mom1snk001'
# TESTKEY = 'FigureC_sep4_mom1src000_mom2src000_mom1snk000'
# TESTKEY = 'FigureHbub_scalar_mom000'
# TESTKEY = 'FigureBub2_mom000'
TESTKEY = 'FigureV_sep3_mom1src000_mom2src000_mom1snk000'
TESTKEY = ''

# Print out the jackknife block at t=TSLICE
# (0..N or ALL for all time slices) for a diagram TESTKEY2
TESTKEY2 = 'FigureV_sep4_mom1src000_mom2src000_mom1snk000'
TESTKEY2 = ''
TSLICE = 0

# ama correction
DOAMA = False
DOAMA = True
#EXACT_CONFIGS = [2050, 2090, 2110, 2240, 2280, 2390, 2410, 2430, 2450, 2470, 1010]
EXACT_CONFIGS = [2050, 2090, 2110, 2240, 2280, 2390, 2410, 2430, 2450, 2470]
EXACT_CONFIGS = [1010, 2410, 2430, 2470]
EXACT_CONFIGS = [ 1130, 1250,  1370]
EXACT_CONFIGS = [1090, 1110, 1130, 1230, 1250,  1370, 1390]

#assert not TEST44, "test option on"
#assert not TEST24C, "test option on"

assert JACKBUB, "not correct.  we need to jackknife the bubbles.  if debugging, comment out"


# Individual diagram's jackknife block to write

def fill_write_block(fndef=FNDEF):
    """compose list of names of the individual diagrams to write"""
    fn1 = h5py.File(fndef, 'r')
    retlist = []
    for i in fn1:
        if 'pioncorrChk' in i:
            spl = i.split('_')
            needed = spl[2:]
            ret = needed[0]
            for j in needed[1:]:
                ret = ret+'_'+j
            retlist.append(ret)
    return retlist

WRITEBLOCK = []
WRITEBLOCK = ['pioncorrChk_mom000']
WRITEBLOCK = fill_write_block(FNDEF)
# only write the single particle correlators
WRITE_INDIVIDUAL = True
WRITE_INDIVIDUAL = False
TDIS_MAX = LT-1 if WRITE_INDIVIDUAL else TDIS_MAX

# debug rows/columns slicing
DEBUG_ROWS_COLS = False

# only save this bubble (speeds up checks involving single bubbles)
BUBKEY = ''
# BUBKEY = 'Figure_Vdis_sep4_mom1000_mom2000'

def getindices(tsep, nmomaux):
    """Get aux indices"""
    if nmomaux == 1:
        retrows = COLS
        retcols = (-ROWST) % LT
    elif nmomaux == 2:
        retrows = COLS
        retcols = (-ROWST-tsep) % LT
    elif nmomaux == 3:
        retrows = np.roll(COLS, -tsep, axis=1)
        retcols = (-ROWST - 2 * tsep) % LT
    return retrows, retcols


@PROFILE
def trajlist(getexactconfigs=False, getsloppysubtraction=False):
    """Get trajectory list from files of form
    <traj>.EXTENSION"""
    assert not getexactconfigs or not getsloppysubtraction, "Must choose one or the other."
    trajl = set()
    suffix = '_exact' if getexactconfigs else ''
    for fn1 in glob.glob(PREFIX+'*'+suffix+'.'+EXTENSION):

        try:
            toadd = int(re.sub(suffix+'.'+EXTENSION, '', re.sub(PREFIX, '', fn1)))
        
        except ValueError:
            if '_exact' not in str(fn1) or getexactconfigs:
                print("skipping:", re.sub(suffix+'.'+EXTENSION, '',
                                          re.sub(PREFIX, '', fn1)))
                print("getexact=", getexactconfigs)
                print("getsloppy=", getsloppysubtraction)
            continue


        # double check if config is sloppy or exact
        if DOAMA:
            check = toadd in EXACT_CONFIGS
            if getsloppysubtraction:
                if not check:
                    continue
            elif getexactconfigs:
                assert check, "found an exact config not in list:"+str(toadd)
            else: # sloppy samples
                if check:
                    continue

        if toadd >= THERMNUM:  # filter out unthermalized
            trajl.add(toadd)
    trajl = sorted(list(trajl))
    if not TEST44 and not TEST24C:
        assert len(trajl)>1, "Len of trajectory list="+str(trajl)
    if getexactconfigs:
        trajl = [str(traj)+'_exact' for traj in trajl]
    if MPIRANK == 0:
        print("Done getting trajectory list, len=", len(trajl))
    return trajl


@PROFILE
def getbasl(fn1):
    """Get base list from selected file"""
    basl = set()
    for dat in fn1:
        #try:
        #    basen = fn1[dat].attrs['basename']
        #except KeyError:
        basen = rf.basename(dat)
        if 'Vdis' not in basen and 'bubble' not in basen:
            #if len(fn1[dat].shape) == 2 and basen:
            basl.add(basen)
    return basl

@PROFILE
def baselist(fn1=None):
    """Get base names of diagrams
    (exclude trajectory info)"""
    if not fn1:
        try:
            fn1 = h5py.File(FNDEF, 'r')
            gn1 = h5py.File(GNDEF, 'r')
            hn1 = h5py.File(HNDEF, 'r')
        except OSError:
            print("Error: unable to locate", FNDEF)
            print("Make sure the working directory is correct.")
            sys.exit(1)
    basl = getbasl(fn1).intersection(getbasl(gn1)).intersection(getbasl(hn1))
    basl_union = getbasl(fn1).union(getbasl(gn1)).union(getbasl(hn1))
    if len(list(basl_union-basl)):
        for i in basl_union-basl:
            assert i in getbasl(hn1), "hn1 missing dataset:"+str(i)
            assert i in getbasl(gn1), "gn1 missing dataset:"+str(i)
            assert i in getbasl(fn1), "fn1 missing dataset:"+str(i)
        if len(list(getbasl(fn1)-getbasl(gn1))):
            print("fn1 larger than gn1")
        elif len(list(getbasl(gn1)-getbasl(fn1))):
            print("gn1 larger than fn1")
        elif len(list(getbasl(hn1)-getbasl(gn1))):
            print("hn1 larger than gn1")
        elif len(list(getbasl(gn1)-getbasl(hn1))):
            print("gn1 larger than hn1")
    assert not basl_union-basl, "Union of basenames is larger than intersection"
    fn1.close()
    gn1.close()
    hn1.close()
    if MPIRANK == 0:
        print("Done getting baselist")
    return basl

@PROFILE
def getbubl(fn1):
    """Get bubble list from selected file"""
    bubl = set()
    for dat in fn1:
        #try:
        #    basen = fn1[dat].attrs['basename']
        #except KeyError:
        basen = rf.basename(dat)
        if 'Vdis' in basen or 'bubble' in basen or 'type' in basen:
            #if len(fn1[dat].shape) == 1 and basen:
            bubl.add(basen)
    return bubl

@PROFILE
def bublist(fn1=None):
    """Get list of disconnected bubbles."""
    if not fn1:
        fn1 = h5py.File(FNDEF, 'r')
        gn1 = h5py.File(GNDEF, 'r')
        hn1 = h5py.File(HNDEF, 'r')
    bubl = getbubl(fn1).intersection(getbubl(gn1)).intersection(getbubl(hn1))
    fn1.close()
    gn1.close()
    hn1.close()
    if MPIRANK == 0:
        print("Done getting bubble list")
    return bubl


@PROFILE
def dojackknife(blk):
    """Apply jackknife to block with shape = (L_traj, L_time)"""
    out = np.zeros(blk.shape, dtype=np.complex)
    if TEST44:
        out = blk
    else:
        for i, _ in enumerate(blk):
            np.mean(np.delete(blk, i, axis=0), axis=0, out=out[i])
    return out


@PROFILE
def h5write_blk(blk, outfile, extension='.jkdat', ocs=None):
    """h5write block.
    """
    outh5 = outfile+extension
    if os.path.isfile(outh5):
        print("File", outh5, "exists. Skipping.")
        return
    print("Writing", outh5, "with", len(blk), "trajectories.")
    assert PRINT_COEFFS, "not printing projection coefficients (bad logging practice)"
    if ocs and PRINT_COEFFS:
        print("Combined Isospin/Subduction coefficients for", outfile, ":")
        try:
            for ctup in ocs[outfile]:
                diagram, coeff = ctup
                print(diagram, ":", coeff)
        except KeyError:
            print(ocs[outfile])
            sys.exit(1)
    filen = h5py.File(outh5, 'w')
    filen[outfile] = blk
    filen.close()
    if MPIRANK == 0:
        print("done writing jackknife blocks: ", outh5)


@PROFILE
def overall_coeffs(iso, irr):
    """Get overall projection coefficients from iso (isopsin coefficients)
    irr (irrep projection)
    """
    ocs = {}
    for iso_dir in iso:
        for operator in irr:
            pol_req = get_polreq(operator)
            mat = re.search(r'I(\d+)/', iso_dir)
            if not mat:
                print("Error: No isopsin info found")
                sys.exit(1)
            isospin_str = mat.group(0)
            opstr = re.sub(isospin_str,
                           '', re.sub(r'sep(\d)+/', '', iso_dir))
            for opstr_chk, outer_coeff in irr[operator]:
                if opstr_chk != opstr:
                    continue
                for original_block, inner_coeff in iso[iso_dir]:
                    pols = rf.pol(original_block)
                    pol_comparison = rf.compare_pols(pols, pol_req)
                    # pol_comparison is false if the polarizations are different
                    # otherwise it is None if there is no polarization
                    # either in pols or pol_req
                    if pol_comparison is not None and not pol_comparison:
                        continue
                    if cross_p(original_block):
                        continue
                    ocs.setdefault(isospin_str+strip_op(operator),
                                   []).append((original_block,
                                               outer_coeff*inner_coeff))
    if MPIRANK == 0:
        print("Done getting projection coefficients")
    return ocs

# obsolete
def cross_p(fname):
    """Check if this is a pipi diagram with cross momenta (x+-x->y+-y)
    """
    mom = rf.mom(fname)
    compare = set()
    ret = False
    return ret
    #if rf.nmom_arr(mom) == 3 and FILTER_OUT_CROSS_MOMENTA:
    #    for i, momex in enumerate(mom):
    #        for j, k in enumerate(momex):
    #            if k != 0:
    #                if i == 0 or not compare:
    #                    compare.add(j)
    #                elif j not in compare:
    #                    ret = True
        

def strip_op(op1):
    """Strips off extra specifiers including polarizations."""
    return op1.split('?')[0]

def get_polreq(op1):
    """Get the required polarization for this irrep, if any
    e.g. T_1_1MINUS needs to have vector particle polarized in x
    """
    spl = op1.split('?')
    if len(spl) == 1:
        reqpol = None
    else:
        polstrspl = spl[1].split(',')[0].split('=')
        if polstrspl[0] == 'pol':
            reqpol = int(polstrspl[1])
        else:
            reqpol = None
    return reqpol


@PROFILE
def jackknife_err(blk):
    """Get jackknife error from block with shape=(L_traj, L_time)"""
    len_t = len(blk)
    avg = np.mean(blk, axis=0)
    prefactor = (len_t-1)/len_t
    err = np.sqrt(prefactor*np.sum((blk-avg)**2, axis=0))
    return avg, err


@PROFILE
def formnum(num):
    """Format complex number in scientific notation"""
    real = '%.8e' % num.real
    if num.imag == 0:
        return real
    else:
        if num.imag < 0:
            plm = ''
        else:
            plm = '+'
        imag = '%.8e' % num.imag
        return real+plm+imag+'j'


@PROFILE
def buberr(bubblks):
    """Show the result of different options for bubble subtraction"""
    for key in bubblks:
        if key == TESTKEY:
            avg, err = jackknife_err(bubblks[key])
            print("Printing first three jackknife samples from t=0:")
            print(bubblks[key][0][0])
            print(bubblks[key][1][0])
            print(bubblks[key][2][0])
            print(key, ":")
            for i, ntup in enumerate(zip(avg, err)):
                avgval, errval = ntup
                print('t='+str(i)+' avg:', formnum(avgval),
                      'err:', formnum(errval))

@PROFILE
def check_match_oplist(ocs):
    """A more stringent check:  Generate the momentum combinations for an operator
    and see if all the diagrams contain these.
    """
    opcheck = opc.op_list('hdf5')
    for irrop in opcheck: # e.g. 'pipisigma_A_1PLUS_mom000'
        irrop_strip = strip_op(irrop)
        print("Checking irrop=", irrop, "with complete diagram listing")
        for opa in ocs: # e.g. 'I0/pipisigma_A_1PLUS_mom000'
            if irrop_strip != opa.split('/')[1]:
                continue
            print("checking momentum combinations of", opa, "with", irrop)
            figlist = set()
            momlist = set()
            diaglist = [tup[0] for tup in ocs[opa]]
            for diag, coeff in ocs[opa]:
                figlist.add(rf.mom_prefix(diag))
            for diag, coeff in opcheck[irrop]:
                momlist.add(rf.mom_suffix(diag))
            for prefix, suffix in itertools.product(figlist, momlist):
                checkstr = prefix + '_' + suffix
                assert checkstr in diaglist, "Missing momentum combination:"+\
                    str(checkstr)+" in op="+str(opa)+"="+str(ocs[opa])+" figlist="+\
                    str(figlist)+" momlist="+str(momlist)


def check_count_of_diagrams(ocs, isospin_str='I0'):
    """count how many diagrams go into irrep/iso projection
    if it does not match the expected, abort
    """
    checks = opc.generateChecksums(isospin_str[-1])
    for opa in list(checks):
        checks[strip_op(opa)] = checks[opa]
        del checks[opa]
    print("checks for I=", isospin_str[-1], "=", checks)
    counter_checks = {}
    isocount = -1
    for irrop in checks: # e.g. 'A_1PLUS_mom000'
        print("irrep checksumming op=", irrop, "for correctness")
        counter_checks[irrop] = 0
        for opa in ocs: # e.g. 'I0/pipisigma_A_1PLUS_mom000'
            if not irrop == getopstr(opa) or isospin_str not in opa:
                continue
            print("isospin checksumming op=", opa, "for correctness")
            if 'rho' in opa:
                if 'pipi' in opa:
                    isocount = 2
                else:
                    isocount = 1
            elif 'sigma' in opa:
                isocount = 2
            else:
                assert 'pipi' in opa, "bad operator:"+str(ocs[opa])
                isocount = 4 if 'I0' in opa else 2
            assert len(ocs[opa]) % isocount == 0, "bad isospin projection count."+str(ocs[opa])+\
                " in "+str(opa)
            print("isocount=", isocount, "number of diagrams in op=", len(ocs[opa]))
            counter_checks[irrop] += len(ocs[opa])/isocount
            print("divided by isocount=", len(ocs[opa])/isocount)
        assert counter_checks[irrop] == checks[irrop], "bad checksum,"+\
            " number of expected diagrams"+\
            " does not match:"+str(checks[irrop])+" vs. "+str(
                counter_checks[irrop])+" "+str(irrop)

def getopstr(opstr):
    """I0/pipisigma_A_1PLUS_mom000->
    A_1PLUS_mom000
    """
    split = opstr.split('_')[1:]
    for i, sub in enumerate(split):
        if 'pipi' in sub or 'sigma' in sub or 'rho' in sub:
            continue
        else:
            split = split[i:]
            break
    ret = ''
    for sub in split:
        ret = ret+sub+'_'
    ret = ret[:-1]
    return ret
    

@PROFILE
def h5sum_blks(allblks, ocs, outblk_shape):
    """Do projection sums on isospin blocks"""
    for opa in ocs:
        mat = re.search(r'(.*)/', opa)
        if mat:
            outdir = mat.group(0)
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
        flag = 0
        ntchk = None
        for base, coeff in ocs[opa]:
            if ntchk is None:
                ntchk = allblks[base].shape[0]
                print('ntchk=', ntchk)
                printt = False
                outblk = np.zeros((ntchk, outblk_shape[1]), dtype=np.complex)
                if ntchk != outblk.shape[0]:
                    print("Warning:", opa,
                          "has different number of trajectories")
                    print("new traj=", ntchk)
                    print("old traj=", outblk_shape[0])
                    print("Be careful in using these blocks in the GEVP!")
                basechk = base
            else:
                if allblks[base].shape[0] != ntchk:
                    # we have a different number of trajectories.
                    # not intra-operator consistent
                    print("Statistics mismatch.  Not enough trajectories.")
                    print("Problematic operator:", opa)
                    print("base:", base, "base check:", basechk)
                    print("number of trajectories in base",
                          allblks[base].shape[0])
                    print("does not match:", ntchk)
                    flag = 1
                    break
            if base == 'FigureC_sep3_mom1src_1_1_1_mom2src111_mom1snk_110' or base == 'FigureC_sep3_mom1src000_mom2src000_mom1snk000':
                temp = fold_time(allblks[base])
                print(base)
                print(allblks[base].shape)
                print(allblks[base])
                printt = True
                for i, row in enumerate(temp):
                    print('traj=', i)
                    for j in row:
                        print(j)
            try:
                outblk += coeff*allblks[base]
            except ValueError:
                # mismatch between shapes of outblk and base.
                # This is a programming error!
                print("Error: trajectory number mismatch")
                print("Problematic operator:", opa)
                print("Problematic base:", base)
                print("allblks[base].shape=", allblks[base].shape)
                print("outblk.shape=", outblk.shape)
                print("coefficient =", coeff)
                print("This is a programming error!  Please fix!")
                flag = 1
                break
        if printt:
            temp = fold_time(outblk)
            print("outblk=", opa)
            print(temp.shape)
            print(temp)
            printt = True
            for i, row in enumerate(temp):
                print('traj=', i)
                for j in row:
                    print(j)
        if flag == 0:
            pass
            h5write_blk(fold_time(outblk), opa, '.jkdat', ocs)
    if MPIRANK == 0:
        print("Done writing summed blocks.")
    return


@PROFILE
def fold_time(outblk, base=''):
    """average data about the midpoint in time for better statistics.
    1/2(f(t)+f(Lt-t))
    """
    if base:
        tsep = rf.sep(base)
        if not tsep:
            tsep = 0
    else:
        tsep = TSEP
    if FOLD:
        if ANTIPERIODIC:
            retblk = [1/2 * (outblk[:, t] - outblk[:, (LT-t-2 * tsep) % LT])
                      for t in range(LT)]
        else:
            retblk = [1/2 * (outblk[:, t] + outblk[:, (LT-t-2*tsep) % LT])
                      for t in range(LT)]
        return np.array(retblk).T
    else:
        return outblk

def get_file_name(traj):
    """Get file name from trajectory"""
    return PREFIX+str(traj)+'.'+EXTENSION


@PROFILE
def getgenconblk(base, trajl, avgtsrc=False, rowcols=None, openlist=None):
    """Get generic connected diagram of base = base
    and indices tsrc, tdis
    """
    rows = None
    cols = None
    if rowcols is not None:
        rows = rowcols[0]
        cols = rowcols[1]
    if avgtsrc:
        blk = np.zeros((len(trajl), LT), dtype=np.complex)
    else:
        blk = np.zeros((len(trajl), LT, LT), dtype=np.complex)
    skip = []
    outarr = np.zeros((LT, LT), dtype=np.complex)
    for i, traj in enumerate(trajl):
        filekey = get_file_name(traj)
        if openlist is None:
            fn1 = h5py.File(filekey, 'r')
        else:
            fn1 = openlist[filekey]
        traj = convert_traj(traj)
        filekey = get_file_name(traj)
        try:
            fn1['traj_'+str(traj)+'_'+base].read_direct(outarr, np.s_[:, :TDIS_MAX+1], np.s_[:, :TDIS_MAX+1])
            #outarr = np.array(fn1['traj_'+str(traj)+'_'+base][:, :TDIS_MAX+1])
        except:
            namec = 'traj_'+str(traj)+'_'+base
            print(namec in fn1)
            print(fn1[namec])
            raise
        #outtemp = np.zeros((LT, LT), dtype=np.complex)
        #outtemp[:, :TDIS_MAX+1] = outarr
        #outarr = outtemp
    #    except KeyError:
    #        skip.append(i)
    #        continue
        if rows is not None and cols is not None:
            outarr = outarr[rows, cols]
        if avgtsrc:
            np.mean(TSTEP*outarr, axis=0, out=blk[i])
        else:
            blk[i] = outarr
    return np.delete(blk, skip, axis=0)


@PROFILE
def getmostblks(basl, trajl, openlist):
    """Get most of the jackknife blocks,
    except for disconnected diagrams"""
    mostblks = {}
    for base in basl:
        if not check_key(base):
            continue
        blk = getgenconblk(base, trajl, avgtsrc=True, openlist=openlist)
        if TESTKEY2:
            print("Printing non-averaged-over-tsrc data")
            printblk(TESTKEY2, blk)
            print("beginning of traj list = ", trajl[0], trajl[1], trajl[2])
        mostblks[base] = dojackknife(blk)
    if MPIRANK == 0:
        print("Done getting most of the jackknife blocks.")
    return mostblks

def convert_traj(traj):
    """convert a string like '1540_exact'
    to int, e.g. 1540
    """
    traj = str(traj)
    return int(re.search(r"(\d)+", traj).group(0))


@PROFILE
def getbubbles(bubl, trajl, openlist=None):
    """Get all of the bubbles."""
    bubbles = {}
    for dsrc in bubl:
        if BUBKEY and dsrc != BUBKEY:
            continue
        for traj in trajl:
            filekey = get_file_name(traj)
            if openlist is None:
                fn1 = h5py.File(filekey, 'r')
            else:
                fn1 = openlist[filekey]
            traj = convert_traj(traj)
            filekey = get_file_name(traj)
            keysrc = 'traj_' + str(traj) + '_' + dsrc
            assert(keysrc in fn1), "key = " + keysrc + \
                " not found in fn1:"+PREFIX+str(traj)+'.'+EXTENSION
            #try:
            #    pdiag = fn1[keysrc].attrs['mom']
            #except KeyError:
            pdiag = rf.mom(keysrc)
            try:
                ptot = rf.ptostr(wd.momtotal(pdiag))
            except:
                print(pdiag, keysrc, filekey)
                sys.exit(1)
            savekey = dsrc+"@"+ptot
            if TAKEREAL and ptot == '000':
                toapp = np.array(fn1[keysrc]).real
            else:
                toapp = np.array(fn1[keysrc])
            bubbles.setdefault(savekey, []).append(toapp)
    for key in bubbles:
        bubbles[key] = np.asarray(bubbles[key])
    #print("Done getting bubbles.")
    return bubbles


@PROFILE
def bubsub(bubbles):
    """Do the bubble subtraction"""
    sub = {}
    for i, bubkey in enumerate(bubbles):
        if i:
            pass
        if NOSUB:
            if JACKBUB:
                sub[bubkey] = np.zeros((len(bubbles[bubkey])))
            else:
                sub[bubkey] = np.zeros((len(bubbles[bubkey][0])))
            continue
        if STILLSUB:
            if bubkey.split('@')[1] != '000':
                sub[bubkey] = np.zeros((len(bubbles[bubkey])))
                continue
        if JACKBUB:
            sub[bubkey] = dojackknife(bubbles[bubkey])
            if TIMEAVGD:
                sub[bubkey] = np.mean(sub[bubkey], axis=1)
        else:
            if TIMEAVGD:
                sub[bubkey] = np.mean(bubbles[bubkey])
            else:
                sub[bubkey] = np.mean(bubbles[bubkey], axis=0)
    #print("Done getting averaged bubbles.")
    return sub


if CONJBUB:
    @PROFILE
    def conjbub(bub):
        """Complex conjugate bubble depending on global variable
        (conjugate)"""
        return np.conjugate(bub)
else:
    @PROFILE
    def conjbub(bub):
        """Complex conjugate bubble depending on global variable
        (do not conjugate)"""
        return bub


def debug_rows(cols, outkey):
    """debug function"""
    if TESTKEY and outkey != TESTKEY:
        return
    if DEBUG_ROWS_COLS:
        print(ROWS)
        print("Now cols")
        print(cols)
        print("Now COLS")
        print(COLS)
        sys.exit(0)


@PROFILE
def getdiscon_name(dsrc_split, dsnk_split):
    """Get output disconnected diagram figure name
    (mimics dataset names of fully connected diagrams)
    """
    ptot = rf.procmom(dsrc_split[1])
    ptot2 = rf.procmom(dsnk_split[1])
    dsrc = dsrc_split[0]
    dsrc = re.sub('type4', 'type4_mom'+rf.ptostr(ptot), dsrc)
    dsnk = dsnk_split[0]
    if not (ptot[0] == -1*ptot2[0] and
            ptot[1] == -1*ptot2[1] and
            ptot[2] == -1*ptot2[2]):  # cc at sink, conserve momentum
        # dummy values to tell the function to stop processing this diagram
        sepval = -1
        discname = None
    else:
        # print(dsrc, dsnk)
        if 'type4' in dsnk:
            sepval = -1
            sepstr = ''
            outfig = None
        else:
            outfig = wd.comb_fig(dsrc, dsnk)
        try:
            sepstr, sepval = wd.get_sep(dsrc, dsnk, outfig)
        except TypeError:
            sepval = -1
            sepstr = ''
        discname = None
        if outfig is not None:
            discname = "Figure"+outfig+sepstr+wd.dismom(rf.mom(dsrc),
                                                        rf.mom(dsnk))
    if discname == TESTKEY:
        print(dsrc, dsnk, sepval)
    return discname, sepval


def check_key(key):
    """Only look at the key in question, tell parent to skip the rest"""
    if (TESTKEY and key != TESTKEY) or (TESTKEY2 and key != TESTKEY2):
        retval = False
    else:
        retval = True
    return retval


@PROFILE
def bubjack(bubl, trajl, openlist, bubbles=None, sub=None):
    """Do jackknife of disconnected (bubble) diagrams"""
    if bubbles is None:
        bubbles = getbubbles(bubl, trajl, openlist=openlist)
    if sub is None:
        sub = bubsub(bubbles)
    return dobubjack(bubbles, sub)

@PROFILE
def dobubjack(bubbles, sub, skipVBub2=False):
    """Now that we have the bubbles,
    compose the diagrams, jackknife
    skipVBub2 if FigureV and FigureBub2 are not needed
    """
    out = {}
    for srckey in bubbles:
        numt = len(bubbles[srckey])
        dsrc_split = srckey.split("@")
        for snkkey in bubbles:
            outkey, sepval = getdiscon_name(dsrc_split, snkkey.split("@"))
            if skipVBub2 and outkey and (
                    'Bub2' in outkey or 'FigureV' in outkey):
                continue
            if sepval < 0 or not check_key(outkey) or outkey is None:
                continue
            cols = np.roll(COLS, -sepval, axis=1)
            debug_rows(cols, outkey)
            out[outkey] = np.zeros((numt, LT), dtype=np.complex)
            if JACKBUB:
                for excl in range(numt):
                    if OUTERSUB:
                        src = np.delete(bubbles[srckey], excl, axis=0)
                        snk = conjbub(np.delete(bubbles[snkkey],
                                                excl, axis=0))
                        outcome = np.tensordot(
                            src, snk, axes=(0, 0))[ROWS, cols]/(
                                len(src)*1.0)-np.outer(
                                    sub[srckey][excl], sub[snkkey][excl])
                    else:
                        src = np.delete(bubbles[srckey],
                                        excl, axis=0)-sub[srckey][excl]
                        snk = conjbub(np.delete(
                            bubbles[snkkey], excl, axis=0)-sub[snkkey][excl])
                        # test 44 is over a single config, so this won't be correct,
                        # but it avoids an error message.
                        outcome = np.tensordot(src, snk, axes=(0, 0))[
                            ROWS, cols]/(len(src)*1.0 if not TEST44 else 1.0)
                        # mean is over tsrc
                        # len(src) division is average over configs
                        # (except for excluded one)
                    np.mean(outcome, axis=0, out=out[outkey][excl])
                    testkey2(outkey, outcome, 0, excl)
            else:
                if OUTERSUB:
                    src = bubbles[srckey]
                    snk = conjbub(bubbles[snkkey])
                    outcome = -1*np.outer(sub[srckey], sub[snkkey])[ROWS, cols]
                    for excl in range(numt):
                        outcome = outcome + np.outer(
                            src[excl], snk[excl])[ROWS, cols]
                        testkey2(outkey, outcome, 0, excl)
                        # np.mean is avg over tsrc
                        np.mean(outcome, axis=0, out=out[outkey][excl])
                else:
                    src = bubbles[srckey]-sub[srckey]
                    snk = conjbub(bubbles[snkkey]-sub[snkkey])
                    for excl in range(numt):
                        outcome = np.outer(src[excl], snk[excl])[ROWS, cols]
                        testkey2(outkey, outcome, 0, excl)
                        # np.mean is avg over tsrc
                        np.mean(outcome, axis=0, out=out[outkey][excl])
                testkey2(outkey, out[outkey], 1)
                out[outkey] = dojackknife(out[outkey])
                testkey2(outkey, out[outkey], 2)
    if MPIRANK == 0:
        print("Done composing disconnected diagrams.")
    return out


def testkey2(outkey, outcome, flag, excl=-1):
    """Print non-averaged over tsrc disconnected diagram"""
    if outkey != TESTKEY2:
        pass
    elif flag == 0 and excl == 0:
        print("Printing non-averaged over" +
              " tsrc disconnected diagram (" +
              " lowest traj number in list):", TESTKEY2)
        print(outcome.shape)
        print(outcome)
    elif flag == 1:
        print("Printing averaged over tsrc disconnected diagram:", TESTKEY2)
        printblk(TESTKEY2, outcome)
    elif flag == 2:
        print("Printing jackknife block for disconnected diagram:", TESTKEY2)
        printblk(TESTKEY2, outcome)
        sys.exit(0)


@PROFILE
def aux_jack(basl, trajl, numt, openlist):
    """Get the aux diagram blocks
    There is a known redundancy with getmostblks:
    a given base may be read, then read again to apply aux.
    This should be eliminated and can save time,
    and could be considered for future improvements
    if a factor of two is needed at the analysis stage.
    """
    auxblks = {}
    for base in basl:
        # get aux diagram name
        outfn = aux.aux_filen(base, stype='hdf5')
        if base in auxblks:
            if AUX_TESTING:
                continue
            else:
                print("aux pair found in data set (redundancy)")
                print("pair =", base, outfn)
                sys.exit(1)
        if not outfn:
            continue
        if TESTKEY and TESTKEY != outfn:
            continue
        if TESTKEY2 and TESTKEY2 != outfn:
            continue
        tsep = rf.sep(base)
        if tsep is not None:
            assert tsep == TSEP, "tsep of base = "+str(tsep)+" base = "+base
        nmomaux = rf.nmom(base)
        # get modified tsrc and tdis
        rows, cols = getindices(tsep, nmomaux)
        # get block from which to construct the auxdiagram
        # mean is avg over tsrc
        blk = np.zeros((numt, LT), dtype=np.complex)
        np.mean(TSTEP*getgenconblk(base, trajl, openlist=openlist,
                                   avgtsrc=False, rowcols=[rows, cols]),
                axis=1, out=blk)
        # now do the jackknife.  apply -1 coefficient if doing aux to a vector T (ccw/cw switch in mirror)
        auxblks[outfn] = dojackknife(blk) *(-1.0 if rf.vecp(base) and 'FigureT' in base else 1.0)
    if MPIRANK == 0:
        print("Done getting the auxiliary jackknife blocks.")
    return auxblks if not NOAUX else {}

@PROFILE
def split_dict_equally(input_dict, chunks=2):
    "Splits dict by keys. Returns a list of dictionaries."
    # prep with empty dicts
    return_list = [{} for _ in range(chunks)]
    idx = 0
    for k,v in input_dict.items():
        return_list[idx][k] = v
        if idx < chunks-1:  # indexes start at 0
            idx += 1
        else:
            idx = 0
    return return_list

@PROFILE
def gatherdicts(gatherblks, root=0):
    """Gather blocks from other sub processes."""
    retdict = {}
    gatherblks = split_dict_equally(gatherblks, chunks=20)
    for dblk in gatherblks:
        dblk = MPI.COMM_WORLD.gather(dblk, root)
        if MPIRANK == root:
            for blkdict in dblk:
                retdict.update(blkdict)
    return retdict

@PROFILE
def getwork(worklistin, mpirank=MPIRANK):
    """Split work over processes."""
    worklist = sorted(list(worklistin))
    work = math.floor(len(worklist)/MPISIZE)
    backfill = len(worklist)-work*MPISIZE
    offset = work*mpirank
    nodework = set(worklist[offset:offset+work])
    baselen = len(nodework)
    if mpirank == 0:
        nodework = nodework.union(set(worklist[work*MPISIZE:]))
        assert len(nodework) == backfill+baselen, "get work bug."
    return nodework

@PROFILE
def getdisconwork(bubl):
    """Get bubble combinations to compose for this rank"""
    bublcomb = set()
    bubl = sorted(list(bubl))
    for src in bubl:
        for snk in bubl:
            bublcomb.add((src, snk))
    nodebublcomb = getwork(list(bublcomb))
    nodebubl = set()
    for src, snk in nodebublcomb:
        nodebubl.add(src)
        nodebubl.add(snk)
    return nodebubl

@PROFILE
def check_inner_outer(ocs, allkeys, auxkeys):
    """Check to make sure the inner pion has energy >= outer pion
    """
    allkeys = set(allkeys)
    auxkeys = set(auxkeys)
    for opa in ocs:
            for diag, _ in ocs[opa]:
                if 'FigureC_' in diag:
                    mom = rf.mom(diag)
                    norm0 = rf.norm2(mom[0])
                    norm1 = rf.norm2(mom[1])
                    assert norm0 >= norm1, "Inner"+\
                        " particle momentum should be >= outer particle"+\
                        " momentum (source). :"+str(diag)
                    mom = np.array(mom)
                    mom3 = mom[0]+mom[1]-mom[2]
                    norm2 = rf.norm2(mom[2])
                    norm3 = rf.norm2(mom3)
                    assert norm2 >= norm3, "Inner"+\
                        " particle momentum should be >= outer particle"+\
                        " momentum (sink). :"+str(diag)
                    assert diag in allkeys, "Missing figure C"+\
                        " from allkeys:"+str(diag)
                    if diag in auxkeys:
                        assert norm0 == norm1, "Inner particle momentum"+\
                            " should be >= outer particle momentum"+\
                            " (source). :"+str(diag)
                        assert norm2 == norm3, "Inner particle momentum"+\
                            " should be >= outer particle momentum"+\
                            " (sink). :"+str(diag)

@PROFILE
def find_unused(ocs, allkeys, auxkeys, fig=None):
    """Find unused diagrams not needed in projection
    """
    fig = 'FigureC' if fig is None else str(fig)
    allkeys = set(allkeys)
    auxkeys = set(auxkeys)
    used = set()
    fn1 = open(fig+'list.txt', 'w')
    try:
        for opa in ocs:
            for diag, _ in ocs[opa]:
                if fig in diag:
                    used.add(diag)
                    assert diag in allkeys, "Missing "+fig+\
                        " from allkeys:"+str(diag)
                    fn1.write(diag+'\n')
    except AssertionError:
        print("missing "+fig+"'s found.  Aborting.")
        sys.exit(1)
    fn1.close()
    print("number of used "+fig+" diagrams in projected set:", len(used))
    allfigc = set()
    for diag in allkeys:
        if fig in diag and diag not in auxkeys:
            allfigc.add(diag)
    print('len=', len(allfigc))
    print('len2=', len(used))
    unused = allfigc.difference(used)
    return unused

@PROFILE
def get_data(getexactconfigs=False, getsloppysubtraction=False):
    """Get jackknife blocks (after this we write them to disk)"""
    bubl = bublist()
    bubl = set() if WRITE_INDIVIDUAL else bubl
    trajl = trajlist(getexactconfigs, getsloppysubtraction)
    basl = baselist()
    if WRITE_INDIVIDUAL:
        basl_new = set()
        for base in basl:
            if base in WRITEBLOCK:
                basl_new.add(base)
        basl = basl_new
    numt = len(trajl)
    openlist = {}
    # this speeds things up but h5py appears to be broken here (gives an error)
    #for traj in trajl:
    #    print("processing traj =", traj, "into memory.")
    #    filekey = PREFIX+str(traj)+'.'+EXTENSION
    #    openlist[filekey] = h5py.File(filekey, 'r', driver='mpio', comm=MPI.COMM_WORLD)
    openlist = None
    # print("done processinge files into memory.")

    # connected work
    nodebases = getwork(basl)
    #disconnected work
    nodebubl = getdisconwork(bubl)

    if not NOAUX:
        ttime = -time.perf_counter()
        auxblks = gatherdicts(aux_jack(nodebases, trajl, numt, openlist))
        ttime += time.perf_counter()
        print("time to get aux blocks:", ttime, "seconds")
    else:
        auxblks = {}

    if MPIRANK == 0:
        if NOAUX:
            assert not auxblks, "Error in NOAUX option.  Non-empty"+\
                " dictionary found of length="+str(len(auxblks))
        else:
            assert auxblks, "Error in NOAUX option.  empty"+\
                " dictionary found"

    ttime = -time.perf_counter()
    mostblks = gatherdicts(getmostblks(nodebases, trajl, openlist))
    ttime += time.perf_counter()
    print("time to get most blocks:", ttime, "seconds")

    ttime = -time.perf_counter()
    bubblks = gatherdicts(bubjack(nodebubl, trajl, openlist))
    ttime += time.perf_counter()
    print("time to get disconnected blocks:", ttime, "seconds")

    if TEST44:
        check_aux_consistency(auxblks, mostblks)


    # do things in this order to overwrite already composed
    # disconnected diagrams (next line)

    if AUX_TESTING:
        allblks = {**mostblks, **auxblks, **bubblks}  # for non-gparity
    else:
        allblks = {**auxblks, **mostblks, **bubblks}  # for non-gparity
    #for filekey in openlist:
    #    openlist[filekey].close()
    return allblks, numt, auxblks

@PROFILE
def check_aux_consistency(auxblks, mostblks):
    """Check consistency of blocks derived via
    aux symmetry with those from production run."""
    count = 0
    exitthis = False
    if MPIRANK == 0:
        for blk in auxblks:
            if blk in mostblks:
                count += 1
                print("unused auxiliary symmetry on diagram:",
                      blk, "total unused=", count)
                try:
                    assert np.allclose(
                        fold_time(auxblks[blk]), fold_time(mostblks[blk]), rtol=1e-08)
                except AssertionError:
                    print("Auxiliary symmetry failure of diagram:"+str(blk))
                    print(fold_time(auxblks[blk]))
                    print('break')
                    print(fold_time(mostblks[blk]))
                    exitthis = True
    exitthis = MPI.COMM_WORLD.bcast(exitthis, 0)
    if exitthis:
        sys.exit(1)
                            

@PROFILE
def check_ama(blknametocheck, sloppyblks, exactblks, sloppysubtractionblks):
    """Check block for consistency across sloppy and exact samples
    return the consistency-checked block lengths
    """
    blk = blknametocheck
    len_exact, lte = exactblks[blk].shape
    len_exact_check, lte_check = sloppysubtractionblks[blk].shape
    len_sloppy, lts = sloppyblks[blk].shape

    # do some checks
    assert len_exact_check == len_exact, "subtraction term in correction \
    has unequal config amount to exact:"+str(len_exact)+", "+str(len_exact_check)
    nexact = len(EXACT_CONFIGS)
    assert len_exact_check == nexact, "subtraction term in correction \
    has unequal config amount to global exact:"+str(nexact)+", "+str(len_exact_check)
    assert lte == lte_check, "subtraction term in correction\
    has unequal time extent to exact:"+str(lte)+", "+str(lte_check)
    assert lte == LT, "exact configs have unequal time extent \
    to global time:"+str(lte)+", "+str(LT)
    assert lts == LT, "sloppy configs have unequal time extent \
    to global time:"+str(lts)+", "+str(LT)

    print("Creating super-jackknife block=", blk,
          "with", nexact, "exact configs and", len_sloppy, "sloppy configs")
    return len_sloppy, len_exact



@PROFILE
def do_ama(sloppyblks, exactblks, sloppysubtractionblks):
    """do ama correction"""
    if not DOAMA or not EXACT_CONFIGS:
        retblks = sloppyblks
    else:
        retblks = {}
        for blk in sloppyblks:

            len_sloppy, len_exact = check_ama(blk, sloppyblks, exactblks,
                                              sloppysubtractionblks)

            # compute correction
            correction = exactblks[blk] - sloppysubtractionblks[blk]

            # create return block (super-jackknife)
            retblks[blk] = np.zeros((len_exact+len_sloppy, LT),
                                    dtype=np.complex)

            sloppy_central_value = np.mean(sloppyblks[blk], axis=0)
            correction_central_value = np.mean(correction, axis=0)

            exact = correction+sloppy_central_value
            sloppy = sloppyblks[blk]+correction_central_value

            retblks[blk][:len_exact] = exact
            retblks[blk][len_exact:] = sloppy

    return retblks


@PROFILE
def main(fixn=True):
    """Run this when making jackknife diagrams from raw hdf5 files"""
    #avg_irreps()
    print('start of main.')
    if not DOAMA:
        allblks, numt, auxblks = get_data()
    else:
        exactblks, numt, auxblks = get_data(True, False)
        sloppyblks, numt, auxblks = get_data(False, False)
        sloppysubtractionblks, numt, auxblks = get_data(False, True)
        allblks = do_ama(sloppyblks, exactblks, sloppysubtractionblks)
    check_diag = "FigureCv3_sep"+str(TSEP)+"_momsrc_100_momsnk000" # sanity check
    check_diag = WRITEBLOCK[0] if WRITE_INDIVIDUAL else check_diag
    if MPIRANK == 0: # write only needs one process, is fast
        assert check_diag in allblks, "sanity check not passing, missing:"+str(check_diag)
        print('check_diag shape=', allblks[check_diag].shape)
    if MPIRANK == 0: # write only needs one process, is fast
        if WRITEBLOCK and not (
                TESTKEY or TESTKEY2):
            for single_block in WRITEBLOCK:
                try:
                    allblks[single_block] = fold_time(allblks[
                        single_block], single_block)
                except KeyError:
                    print(single_block, "not found.  not writing.")
                    sys.exit(1)
                h5write_blk(allblks[single_block],
                            single_block, extension='.jkdat', ocs=None)
        if not WRITE_INDIVIDUAL:
            # allblks = {**mostblks, **bubblks} # for gparity
            ocs = overall_coeffs(
                isoproj(fixn, 0, dlist=list(
                    allblks.keys()), stype=STYPE), opc.op_list(stype=STYPE))
            # do a checksum to make sure we have all the diagrams we need
            for i in ocs:
                print(i)
            check_count_of_diagrams(ocs, "I0")
            check_count_of_diagrams(ocs, "I2")
            check_count_of_diagrams(ocs, "I1")
            check_match_oplist(ocs)
            check_inner_outer(
                ocs, allblks.keys() | set(), auxblks.keys() | set())
            unused = set()
            for fig in ['FigureR' 'FigureC' 'FigureD',
                        # 'FigureV', 'FigureCv3', 'FigureCv3R',
                        'FigureBub2', 'FigureT']:
                unused = find_unused(
                    ocs, allblks.keys() | set(),
                    auxblks.keys() | set(), fig=fig).union(unused)
            count = len(unused)
            for useless in sorted(list(unused)):
                try:
                    if not (rf.vecp(useless) and rf.norm2(wd.momtotal(rf.mom(useless)))): # we don't analyze all of pcom for I=1 yet
                        if not rf.checkp(useless) and not (rf.vecp(useless) and rf.allzero(rf.mom(useless))):
                            print("unused diagram:", useless)
                            count -= 1
                except TypeError:
                    print(useless)
                    sys.exit(1)
            print("length of unused=", len(unused))
            assert len(unused) == 0 or count == len(unused), "Unused (non-vec) diagrams exist."
            if TESTKEY:
                buberr(allblks)
                sys.exit(0)
            h5sum_blks(allblks, ocs, (numt, LT))
            avg_irreps()

@PROFILE
def avg_irreps():
    """Average irreps"""
    if MPIRANK == 0:
        for isostr in ('I0', 'I1', 'I2'):
            for irrep in AVG_ROWS:
                op_list = set()
                # extract the complete operator list from all the irrep rows
                for example_row in AVG_ROWS[irrep]:
                    for op1 in glob.glob(isostr+'/'+'*'+example_row+\
                                         '.jkdat'):
                        op_list.add(re.sub(example_row+'.jkdat',
                                           '', re.sub(isostr+'/', '', op1)))
                for op1 in list(op_list):
                    avg_hdf5.OUTNAME = isostr+'/'+op1+irrep+'.jkdat'
                    avg_list = []
                    for row in AVG_ROWS[irrep]:
                        avg_list.append(isostr+'/'+op1+row+'.jkdat')
                    avg_hdf5.main(*avg_list)

@PROFILE
def printblk(basename, blk):
    """Print jackknife block (testing purposes)"""
    if isinstance(TSLICE, int):
        print("Printing time slice", TSLICE, "of", basename)
        print(blk.shape)
        print(blk[:, TSLICE])
    else:
        print("Printing all time slices of", basename)
        print(blk)


if __name__ == '__main__':
    FIXN = None
    if MPIRANK == 0:
        # FIXN = input("Need fix norms before summing? True/False?")
        # FIXN = 'False'
        FIXN = '0'
        FIXNSTR = FIXN
        FIXN = FIXN in ['true', '1', 't', 'y',
                        'yes', 'yeah', 'yup', 'certainly', 'True']
        if not FIXN and FIXNSTR not in ['false', '0', 'f', 'n',
                                        'no', 'nope', 'nah',
                                        'certainly not', 'False']:
            sys.exit(1)
    # FIXN = MPI.COMM_WORLD.bcast(FIXN, 0)
    FIXN = False
    START = time.perf_counter()
    main(FIXN)
    END = time.perf_counter()
    if MPIRANK == 0:
        print("Total elapsed time =", END-START, "seconds")
