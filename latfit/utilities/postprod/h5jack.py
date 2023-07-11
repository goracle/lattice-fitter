#!/usr/bin/python3
"""Write jackknife blocks from h5py files"""
import sys
import time
import os
import re
import glob
import math
import pickle
import subprocess
import ast
import numpy as np
from mpi4py import MPI
import h5py

# static
from latfit.utilities import op_compose as opc
from latfit.utilities import exactmean as em
from latfit.utilities.postprod.auxblks import NOAUX, AUX_TESTING
from latfit.utilities import read_file as rf
from latfit.utilities.op_compose import AVG_ROWS
from latfit.utilities.sum_blks import isoproj
from latfit.utilities.postprod.checkblks import strip_op
from latfit.utilities.postprod.checkblks import TESTKEY, TESTKEY2
from latfit.utilities.postprod.checkblks import FREEFIELD, TEST44, TEST24C

# dynamic
import latfit.utilities.postprod.auxblks as auxb
import latfit.utilities.postprod.bubblks as bubb
import latfit.utilities.postprod.mostblks as mostb
import latfit.utilities.postprod.checkblks as checkb
import latfit.utilities.avg_hdf5 as avg_hdf5

# when writing pion correlators, average over tsrc or leave un-averagd
AVGTSRC = False
AVGTSRC = True

# only write the single particle correlators
WRITE_INDIVIDUAL = True
WRITE_INDIVIDUAL = False

# ama correction
DOAMA = False
DOAMA = True

# size of lattice in time, lattice units
LT = 64 if not TEST44 else 4
LT = LT if not TEST24C else 64
LT = 32 if FREEFIELD else LT
#TSEP = 3 if not TEST44 else 1
#TSEP = TSEP if not TEST24C else 3
#TSEP = 4 if not TEST44 else 1
#TSEP = 4 if FREEFIELD else TSEP

# other config options
THERMNUM = 0  # eliminate configs below this number to thermalize
THERMNUM = THERMNUM if not TEST44 else 4539  # eliminate configs below this number to thermalize
THERMNUM = THERMNUM if not TEST24C else 0
#TSTEP = 64/6 if not TEST44 else 1  # we only measure every TSTEP time slices to save on time
#TSTEP = 8 if not TEST44 else 1  # we only measure every TSTEP time slices to save on time
#TSTEP = TSTEP if not TEST24C else 8

# max time distance between (inner) particles
#TDIS_MAX = 64
#TDIS_MAX = 22
#TDIS_MAX = 16

LATTICE_ENSEMBLE = '32c'
LATTICE_ENSEMBLE = '24c'
ENSEMBLE_DICT = {}
ENSEMBLE_DICT['32c'] = {'tsep' : 4, 'tdis_max' : 22, 'tstep' : 64/6}
ENSEMBLE_DICT['24c'] = {'tsep' : 3, 'tdis_max' : 16, 'tstep' : 8}
if TEST44:
    for i in ENSEMBLE_DICT:
        ENSEMBLE_DICT[i] = {'tsep' : 1, 'tdis_max' : 4, 'tstep' : 2}
#### RARELY MODIFY (AUTOFILLED OR OBSOLETE)

def tdismax():
    """Return tdis max"""
    if WRITE_INDIVIDUAL:
        ret = LT-1
    else:
        ret = ENSEMBLE_DICT[LATTICE_ENSEMBLE]['tdis_max']
    return ret
mostb.tdismax = tdismax

# FOLD = True # average about the mirror point in time (True)
FOLD = True
# Print isospin and irrep projection coefficients of operator to be written
PRINT_COEFFS = True
# ANTIPERIODIC in time (false for mesons,
# the true option only happens for tanh shapes of the correlator);
# this is a bad feature, keep it false!
ANTIPERIODIC = False

# skip all the vector diagrams (if the I=1 data is missing)
SKIP_VEC = True
SKIP_VEC = False

# check for KK operator
KK_OP = True
KK_OP = False

# check for kaon->kaon correlator
KCORR = True
KCORR = False

# representative hdf5 file, to get info about lattice
PREFIX = 'traj_'
EXTENSION = 'hdf5'
# format for files; don't change
STYPE = 'hdf5'


FNDEF = PREFIX+'250.'+EXTENSION
GNDEF = PREFIX+'250.'+EXTENSION
HNDEF = PREFIX+'250.'+EXTENSION
FNDEF = PREFIX+'270.'+EXTENSION
GNDEF = PREFIX+'280.'+EXTENSION
HNDEF = PREFIX+'290.'+EXTENSION
if TEST44:
    FNDEF = PREFIX+'4540.'+EXTENSION
    GNDEF = PREFIX+'4540.'+EXTENSION
    HNDEF = PREFIX+'4540.'+EXTENSION
if TEST24C:
    FNDEF = PREFIX+'2460.'+EXTENSION
    GNDEF = PREFIX+'2460.'+EXTENSION
    HNDEF = PREFIX+'2460.'+EXTENSION
if FREEFIELD:
    FNDEF = PREFIX+'1000.'+EXTENSION
    GNDEF = PREFIX+'1000.'+EXTENSION
    HNDEF = PREFIX+'1000.'+EXTENSION


#### set variables in submodules, DO NOT MODIFY BELOW THIS LINE

# precomputed indexing matrices; DON'T CHANGE
ROWST = np.tile(np.arange(LT), (LT, 1))
ROWS = np.tile(np.arange(LT), (LT, 1)).T
COLS = np.array([np.roll(np.arange(LT), -i, axis=0) for i in range(LT)])

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()

EXACT_CONFIGS = []
if DOAMA:
    for FN1 in glob.glob(PREFIX+'*'+'_exact'+'.'+EXTENSION):
        TOADD = int(re.sub('_exact'+'.'+EXTENSION, '',
                           re.sub(PREFIX, '', FN1)))
        EXACT_CONFIGS.append(TOADD)
print("Configs with sloppy and exact versions:", EXACT_CONFIGS)

WRITEBLOCK = []
if AVGTSRC:
    WRITEBLOCK = ['pioncorrChk_mom000']
    WRITEBLOCK = ['kaoncorrChk_mom000']
else:
    WRITEBLOCK = ['pioncorr_mom000']
    WRITEBLOCK = ['pioncorrChk_mom000']
    WRITEBLOCK = ['kaoncorrChk_mom000']
AVGTSRC = True if not WRITE_INDIVIDUAL else AVGTSRC

# set aux variables
auxb.TSEP = ENSEMBLE_DICT[LATTICE_ENSEMBLE]['tsep']
auxb.LT = LT
auxb.TSTEP = ENSEMBLE_DICT[LATTICE_ENSEMBLE]['tstep']

# dynamically set bub variables
bubb.TSEP = ENSEMBLE_DICT[LATTICE_ENSEMBLE]['tsep']
bubb.LT = LT
bubb.TSTEP = ENSEMBLE_DICT[LATTICE_ENSEMBLE]['tstep']
bubb.ROWS = ROWS
bubb.COLS = COLS
bubb.KK_OP = KK_OP # check for KK op if true.

# dynamically set mostblks variables
mostb.TSEP = ENSEMBLE_DICT[LATTICE_ENSEMBLE]['tsep']
mostb.LT = LT
mostb.TSTEP = ENSEMBLE_DICT[LATTICE_ENSEMBLE]['tstep']
mostb.ROWS = ROWS
mostb.COLS = COLS
mostb.ROWST = ROWST
mostb.PREFIX = PREFIX
mostb.EXTENSION = EXTENSION
mostb.WRITE_INDIVIDUAL = WRITE_INDIVIDUAL
mostb.TDIS_MAX = ENSEMBLE_DICT[LATTICE_ENSEMBLE]['tdis_max']
mostb.AVGTSRC = AVGTSRC
mostb.KK_OP = KK_OP
mostb.KCORR = KCORR

# check blks dynamic
checkb.LT = LT
checkb.SKIP_VEC = SKIP_VEC
checkb.EXACT_CONFIGS = EXACT_CONFIGS
checkb.AVGTSRC = AVGTSRC
checkb.ENSEMBLE_DICT = ENSEMBLE_DICT
checkb.DOAMA = DOAMA
checkb.SKIP_VEC = SKIP_VEC
checkb.WRITE_INDIVIDUAL = WRITE_INDIVIDUAL

def check_ids(ensemble):
    """Import hack"""
    return checkb.check_ids(ensemble)

# Individual diagram's jackknife block to write

#### list filling

def fill_fndefs():
    """Fill in the trajectories to bootstrap diagram list from"""
    alist = []
    for i in glob.glob(PREFIX+'*'+'.'+EXTENSION):
        alist.append(i)
    alist = sorted(alist)
    if TEST44:
        alist = [alist[0], alist[0], alist[0]]
    return (alist[0], alist[1], alist[2])

def fill_write_block(fndef=FNDEF):
    """compose list of names of the individual diagrams to write"""
    fn1 = h5py.File(fndef, 'r')
    retlist = []
    for i in fn1:
        if AVGTSRC:
            cond = 'pioncorrChk' in i
            cond = 'kaoncorrChk' in i
        else:
            cond = 'pioncorr' in i and 'Chk' not in i
            cond = 'pioncorrChk' in i
            cond = 'kaoncorr' in i and 'Chk' not in i
            cond = 'kaoncorrChk' in i
        if cond:
            spl = i.split('_')
            needed = spl[2:]
            ret = needed[0]
            for j in needed[1:]:
                ret = ret+'_'+j
            retlist.append(ret)
    retlist = sorted(retlist)
    return retlist

@PROFILE
def trajlist(getexactconfigs=False, getsloppysubtraction=False):
    """Get trajectory list from files of form
    <traj>.EXTENSION"""
    assert not getexactconfigs or not getsloppysubtraction, \
        "Must choose one or the other."
    trajl = set()
    suffix = '_exact' if getexactconfigs else ''
    globex = glob.glob(PREFIX+'*'+suffix+'.'+EXTENSION)
    lenexact = len(globex) if getexactconfigs else len(EXACT_CONFIGS)
    assert len(EXACT_CONFIGS) == lenexact
    for fn1 in globex:

        try:
            toadd = int(re.sub(suffix+'.'+EXTENSION, '',
                               re.sub(PREFIX, '', fn1)))
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
                assert check, "found an exact config not in list:"+str(
                    toadd)
            else: # sloppy samples
                if check:
                    continue

        if toadd >= THERMNUM:  # filter out unthermalized
            trajl.add(toadd)
    trajl = sorted(list(trajl))
    if not TEST44 and not TEST24C and not FREEFIELD:
        assert len(trajl) > 1, "Len of trajectory list="+str(trajl)
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
def base_list(fn1=None):
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
    if sorted(list(basl_union-basl)):
        for i in basl_union-basl:
            assert i in getbasl(hn1), "hn1 missing dataset:"+str(i)
            assert i in getbasl(gn1), "gn1 missing dataset:"+str(i)
            assert i in getbasl(fn1), "fn1 missing dataset:"+str(i)
        if sorted(list(getbasl(fn1)-getbasl(gn1))):
            print("fn1 larger than gn1")
        elif sorted(list(getbasl(gn1)-getbasl(fn1))):
            print("gn1 larger than fn1")
        elif sorted(list(getbasl(hn1)-getbasl(gn1))):
            print("hn1 larger than gn1")
        elif sorted(list(getbasl(gn1)-getbasl(hn1))):
            print("gn1 larger than hn1")
    assert not basl_union-basl, "Union of basenames is larger than intersection"
    fn1.close()
    gn1.close()
    hn1.close()
    if MPIRANK == 0:
        print("Done getting baselist")
    basl = sorted(list(basl))
    return basl

def individual_bases(basl):
    """Get individual jackknife block names to be written (no projection)"""
    assert WRITE_INDIVIDUAL, "switch mismatch, bug."
    basl_new = set()
    for base in basl:
        if base in WRITEBLOCK:
            basl_new.add(base)
    basl = basl_new
    basl = sorted(list(basl))
    return basl

def prune_vec(baselist):
    """Get rid of all I=1 (if skipping that in production)
    """
    ret = set()
    for base in baselist:
        if rf.vecp(base) or 'vecCheck' in base:
            continue
        ret.add(base)
    ret = sorted(list(ret))
    return ret

def prune_nonequal_crosspol(baselist):
    """Get rid of vec-vec with non equal src snk polarizations
    (from an earlier version of the production)
    """
    ret = set()
    for base in baselist:
        pol = rf.pol(base)
        if not isinstance(pol, int) and pol is not None:
            assert isinstance(pol[0], int)
            if pol[0] != pol[1]:
                continue
        ret.add(base)
    ret = sorted(list(ret))
    return ret

#### write

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
        for base, coeff, _ in ocs[opa]:
            if ntchk is None:
                ntchk = allblks[base].shape[0]
                print('ntchk=', ntchk)
                printt = False
                outsum = []
                outblk = np.zeros((
                    ntchk, outblk_shape[1]), dtype=np.complex)
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
            if base in ('FigureC_sep3_mom1src_1_1_1_mom2src111_mom1snk_110',
                        'FigureC_sep3_mom1src000_mom2src000_mom1snk000'):
                printt = checkb.debugprint(allblks[base], base)
            try:
                outsum.append(coeff*fold_time(allblks[base], base))
                #outblk += coeff*allblks[base]
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
        assert np.all(outblk == 0.0), "out block not zero'd"
        outblk += em.acsum(np.asarray(outsum, dtype=np.complex128))
        if printt:
            printt = checkb.debugprint(outblk, '', pstr='outblk=')
        if flag == 0:
            h5write_blk(outblk, opa, '.jkdat', ocs)
    if MPIRANK == 0:
        print("Done writing summed blocks.")

@PROFILE
def h5write_blk(blk, outfile, extension='.jkdat', ocs=None):
    """h5write block.
    """
    extension = '_unsummed'+extension if not AVGTSRC else extension
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
                diagram, coeff, _ = ctup
                print(diagram, ":", coeff)
        except KeyError:
            print(ocs[outfile])
            sys.exit(1)
    filen = h5py.File(outh5, 'w')
    filen[outfile] = blk
    filen.close()
    if MPIRANK == 0:
        print("done writing jackknife blocks: ", outh5)

### coefficient


def nnon0(vec):
    """Count number of non-zero components"""
    assert hasattr(vec, '__iter__')
    ret = 0
    for i in vec:
        if i:
            ret += 1
    return ret

def polcoeff(original_block, pol_coeffs):
    """ Get the particular polarization coefficient we need
    """
    pols = rf.pol(original_block)
    polcount = 0
    if pols is not None and np.asarray(pol_coeffs).shape:
        polcount = nnon0(pol_coeffs)
        try:
            # -1 because of 0 indexing
            # not being applied to polarizations
            pol_coeff = pol_coeffs[
                rf.firstpol(original_block)-1]
        except IndexError:
            print("bad block cause:", original_block)
            print("pol coefficients:", pol_coeffs)
            raise
    elif pols is None:
        pol_coeff = 1.0
    else:
        assert not np.asarray(
            pol_coeffs).shape, str(pol_coeffs)
        assert pol_coeffs == 1.0,\
            "bad polarization coeff:"+str(pol_coeffs)
        pol_coeff = pol_coeffs
    return pol_coeff, polcount

@PROFILE
def overall_coeffs(iso, irr):
    """Get overall projection coefficients from iso (isopsin coefficients)
    irr (irrep projection)
    """
    ocs = {}
    print("getting overall projection coefficients")
    for operator in irr:
        pol_req, pol_coeffs = get_polreq(operator)
        for iso_dir in iso:
            isospin_str = re.search(r'I(\d+)/', iso_dir).group(0)
            for opstr_chk, outer_coeff in irr[operator]:
                if opstr_chk != re.sub(
                        isospin_str, '', re.sub(r'sep(\d)+/', '', iso_dir)):
                    continue
                for original_block, inner_coeff in iso[iso_dir]:
                    pol_comparison = rf.compare_pols(rf.pol(original_block),
                                                     pol_req)
                    if pol_comparison is not None and not pol_comparison:
                        continue
                    pol_coeff, polcount = polcoeff(original_block,
                                                   pol_coeffs)
                    #if not pol_coeff*outer_coeff*inner_coeff:
                    #continue
                    ocs.setdefault(
                        isospin_str+strip_op(operator),
                        []).append(
                            (original_block,
                             pol_coeff*outer_coeff*inner_coeff, polcount))
    if MPIRANK == 0:
        print("Done getting projection coefficients")
    return ocs

def get_polreq(op1):
    """Get the required polarization for this irrep, if any
    e.g. T_1_1MINUS needs to have vector particle polarized in x
    """
    spl = op1.split('?')
    pol_coeffs = np.array(1.0)
    if len(spl) == 1:
        reqpol = None
    else:
        polstrspl = spl[1].split('=')
        if polstrspl[0] == 'pol':
            try:
                reqpol = int(polstrspl[1])
            except ValueError:
                reqpol = None
                try:
                    pol_coeffs = np.asarray(ast.literal_eval(polstrspl[1]),
                                            dtype=complex)
                    pol_coeffs = pol_sq_conj(op1, pol_coeffs)
                except SyntaxError:
                    print('unable to parse:', polstrspl[1], 'as code.')
                    print("op1 =", op1)
                    print("spl =", spl)
                    print('polstrspl =', polstrspl)
                    raise
                assert len(pol_coeffs) == 3, \
                    "need 3 coefficients for each polarization"
        else:
            reqpol = None
    return reqpol, pol_coeffs


def pol_sq_conj(op1, pol_coeffs):
    """square/conjugate pol. coefficients as needed"""
    if 'rhorho' in op1:
        ret = pol_coeffs*np.conj(pol_coeffs)
    elif 'pipirho' in op1:
        ret = np.conj(pol_coeffs)
    else:
        ret = pol_coeffs
    return ret


#### averaging, jackknifing

@PROFILE
def dojackknife(blk):
    """Apply jackknife to block with shape = (L_traj, L_time)"""
    out = np.zeros(blk.shape, dtype=complex)
    if len(blk) == 1:
        out = blk
    else:
        if TEST44:
            out = blk
        else:
            assert len(blk) > 1, \
                "block length should be greater than 1 for jackknife"
            for i, _ in enumerate(blk):
                out[i] = em.acmean(np.delete(blk, i, axis=0))
    return out
auxb.dojackknife = dojackknife
bubb.dojackknife = dojackknife
mostb.dojackknife = dojackknife

@PROFILE
def fold_time(outblk, base, dofold=False):
    """average data about the midpoint in time for better statistics.
    1/2(f(t)+f(Lt-t))
    """
    if base:
        tsep = rf.sep(base)
        if not tsep:
            tsep = 0
    else:
        tsep = ENSEMBLE_DICT[LATTICE_ENSEMBLE]['tsep']
    if FOLD and AVGTSRC:
        retblk = []
        mult = 2 if 'Cv3' not in base and 'FigureT' not in base else 1
        if 'Bub2' in base or 'FigureV' in base or 'Cv3' in base or 'VKK' in base:
            dofold = True
        elif 'corr' in base and WRITE_INDIVIDUAL:
            dofold = True
        if dofold:
            print("folding base:", base, "with tsep =", tsep)
        for t in range(LT):
            if dofold:
                assert np.all(outblk[:, (
                    LT-t-mult*tsep) % LT]), (
                        outblk[:, (LT-t-mult*tsep) % LT],
                        t, mult, base, outblk[:, t], outblk.shape, (LT-t-mult*tsep) % LT)
                if ANTIPERIODIC:
                    retblk.append(1/2 * (outblk[:, t] - outblk[
                        :, (LT-t-mult * tsep) % LT]))
                else:
                    retblk.append(1/2 * (outblk[:, t] + outblk[
                        :, (LT-t-mult*tsep) % LT]))
            else:
                retblk.append(outblk[:, t])
        assert len(retblk) == LT
        ret = np.array(retblk).T
    else:
        ret = outblk
    return ret
auxb.fold_time = fold_time
bubb.fold_time = fold_time
mostb.fold_time = fold_time
checkb.fold_time = fold_time

@PROFILE
def do_ama(sloppyblks, exactblks, sloppysubtractionblks):
    """do ama correction"""
    if not DOAMA or not EXACT_CONFIGS:
        retblks = sloppyblks
    else:
        retblks = {}
        for blk in sloppyblks:

            if not KK_OP and ('KK2KK' in blk or 'KK2sigma' in blk or 'sigma2KK' in blk or 'KK2pipi' in blk or 'pipi2KK' in blk):
                assert KK_OP is not None, "set KK_OP in h5jack.py"
                continue
            if not KCORR and ('kaoncorr' in blk or 'Hbub_kaon' in blk):
                assert KCORR is not None, "set KCORR in h5jack.py"
                continue

            len_sloppy, len_exact = checkb.check_ama(blk, sloppyblks,
                                                     exactblks,
                                                     sloppysubtractionblks)

            # compute correction
            checkb.check_dup_configs([exactblks[blk]])
            checkb.check_dup_configs([sloppysubtractionblks[blk]])
            correction = exactblks[blk] - sloppysubtractionblks[blk]
            checkb.check_dup_configs([correction])

            # create return block (super-jackknife)
            shape = (len_exact+len_sloppy, LT) if AVGTSRC else (
                len_exact+len_sloppy, LT, LT)
            retblks[blk] = np.zeros(shape, dtype=np.complex)

            sloppy_central_value = em.acmean(sloppyblks[blk])
            correction_central_value = em.acmean(correction)

            exact = correction+sloppy_central_value
            sloppy = sloppyblks[blk]+correction_central_value

            retblks[blk][:len_exact] = exact
            retblks[blk][len_exact:] = sloppy

    return retblks

#### multiprocessing

@PROFILE
def split_dict_equally(input_dict, chunks=2):
    "Splits dict by keys. Returns a list of dictionaries."
    # prep with empty dicts
    return_list = [{} for _ in range(chunks)]
    idx = 0
    for k, vitem in input_dict.items():
        return_list[idx][k] = vitem
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

def get_file_name(traj):
    """Get file name from trajectory"""
    return PREFIX+str(traj)+'.'+EXTENSION
bubb.get_file_name = get_file_name
mostb.get_file_name = get_file_name

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
bubb.getwork = getwork
mostb.getwork = getwork

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
bubb.getindices = getindices
auxb.getindices = getindices
mostb.getindices = getindices



@PROFILE
def get_data(getexactconfigs=False, getsloppysubtraction=False):
    """Get jackknife blocks (after this we write them to disk)"""
    bubl = bubb.bublist()
    bubl = [] if WRITE_INDIVIDUAL else bubl
    trajl = trajlist(getexactconfigs, getsloppysubtraction)
    basl = base_list()
    basl = individual_bases(basl) if WRITE_INDIVIDUAL else basl
    basl = prune_nonequal_crosspol(basl)
    if SKIP_VEC:
        basl = prune_vec(basl)
    openlist = {}
    # this speeds things up but
    # h5py appears to be broken here (gives an error)
    #for traj in trajl:
    #    print("processing traj =", traj, "into memory.")
    #    filekey = PREFIX+str(traj)+'.'+EXTENSION
    #    openlist[filekey] = h5py.File(
    # filekey, 'r', driver='mpio', comm=MPI.COMM_WORLD)
    openlist = None
    # print("done processinge files into memory.")

    # connected work
    nodebases = getwork(basl)
    #disconnected work
    nodebubl = bubb.getdisconwork(bubl)

    if not NOAUX:
        ttime = -time.perf_counter()
        auxblks = gatherdicts(auxb.aux_jack(nodebases, trajl,
                                            len(trajl), openlist))
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
    bubblks = gatherdicts(bubb.bubjack(nodebubl, trajl, openlist))
    ttime += time.perf_counter()
    print("time to get disconnected blocks:", ttime, "seconds")

    ttime = -time.perf_counter()
    mostblks = gatherdicts(mostb.getmostblks(nodebases, trajl, openlist))
    ttime += time.perf_counter()
    print("time to get most blocks:", ttime, "seconds")

    if TEST44:
        auxb.check_aux_consistency(auxblks, mostblks)


    # do things in this order to overwrite already composed
    # disconnected diagrams (next line)

    if AUX_TESTING:
        allblks = {**mostblks, **auxblks, **bubblks}  # for non-gparity
    else:
        allblks = {**auxblks, **mostblks, **bubblks}  # for non-gparity
    #for filekey in openlist:
    #    openlist[filekey].close()
    return allblks, len(trajl), auxblks


@PROFILE
def main(fixn=False):
    """Run this when making jackknife diagrams from raw hdf5 files"""
    print('start of main.')
    check_diag = "FigureCv3_sep"+str(
        ENSEMBLE_DICT[LATTICE_ENSEMBLE]['tsep'])+"_momsrc_100_momsnk000" # sanity check
    check_diag = WRITEBLOCK[0] if WRITE_INDIVIDUAL else check_diag
    if not DOAMA:
        allblks, numt, auxblks = get_data()
        checkb.check_dup_configs(allblks)
    else:
        exactblks, numt, auxblks = get_data(True, False)
        checkb.check_dup_configs(exactblks)
        # write only needs one process, is fast
        if MPIRANK == 0 and not (TESTKEY or TESTKEY2):
            assert check_diag in exactblks, \
                "sanity check not passing, missing:"+str(check_diag)
            print('check_diag shape=', exactblks[check_diag].shape)
        sloppyblks, numt, auxblks = get_data(False, False)
        checkb.check_dup_configs(sloppyblks)
        sloppysubtractionblks, numt, auxblks = get_data(False, True)
        checkb.check_dup_configs(sloppysubtractionblks)
        allblks = do_ama(sloppyblks, exactblks, sloppysubtractionblks)
        checkb.check_dup_configs(allblks)
    # write only needs one process, is fast
    if MPIRANK == 0 and not (TESTKEY or TESTKEY2):
        assert check_diag in allblks, \
            "sanity check not passing, missing:"+str(check_diag)
        print('check_diag shape=', allblks[check_diag].shape)
    if MPIRANK == 0: # write only needs one process, is fast
        if WRITEBLOCK and not (
                TESTKEY or TESTKEY2):
            for single_block in WRITEBLOCK:
                if not KK_OP and ('KK2KK' in single_block or 'KK2sigma' in single_block or 'sigma2KK' in single_block or 'KK2pipi' in single_block or 'pipi2KK' in single_block):
                    assert KK_OP is not None, "set KK_OP in h5jack.py"
                    continue
                if not KCORR and ('kaoncorr' in single_block or 'Hbub_kaon' in single_block):
                    assert KCORR is not None, "set KCORR in h5jack.py"
                    continue
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
                isoproj(fixn, 0, dlist=sorted(list(
                    allblks.keys())), stype=STYPE), opc.op_list(
                        stype=STYPE))
            checkb.checksum_diagrams(ocs, allblks, auxblks)
            if TESTKEY:
                print("displaying block:", TESTKEY)
                bubb.buberr(allblks)
                sys.exit(0)
            else:
                h5sum_blks(allblks, ocs, (numt, LT))
                avg_irreps()

@PROFILE
def avg_irreps(ext='.jkdat'):
    """Average irreps"""
    if MPIRANK == 0:
        for isostr in ('I0', 'I1', 'I2'):
            for irrep in AVG_ROWS:
                op_list = set()
                # extract the complete operator list from all the irrep rows
                for example_row in AVG_ROWS[irrep]:
                    for op1 in glob.glob(isostr+'/'+'*'+example_row+\
                                         ext):
                        op_list.add(re.sub(example_row+ext,
                                           '', re.sub(isostr+'/', '', op1)))
                for op1 in sorted(list(op_list)):
                    avg_hdf5.OUTNAME = isostr+'/'+op1+irrep+ext
                    avg_list = []
                    for row in AVG_ROWS[irrep]:
                        avg_list.append(isostr+'/'+op1+row+ext)
                    avg_hdf5.main(*avg_list)

if __name__ == '__main__':
    try:
        open('ids.p', 'rb')
    except FileNotFoundError:
        print("be sure to set TSEP, TDIS_MAX, TSTEP, DOAMA, SKIP_VEC",
              "correctly")
        print("edit h5jack.py to remove the hold then rerun")
        # the hold
        sys.exit(1)
        IDS = [ENSEMBLE_DICT[LATTICE_ENSEMBLE]['tsep'],
               ENSEMBLE_DICT[LATTICE_ENSEMBLE]['tdis_max'],
               ENSEMBLE_DICT[LATTICE_ENSEMBLE]['tstep'], DOAMA, SKIP_VEC]
        IDS = np.asarray(IDS)
        pickle.dump(IDS, open('ids.p', "wb"))

    checkb.check_ids(LATTICE_ENSEMBLE)

    FNDEF, GNDEF, HNDEF = fill_fndefs()
    # dynamically set
    bubb.FNDEF = FNDEF
    bubb.GNDEF = GNDEF
    bubb.HNDEF = HNDEF
    WRITEBLOCK = fill_write_block(FNDEF)
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
        subprocess.check_output(['notify-send', '-u', 'critical',
                                 '-t', '30', "h5jack: local run complete"])

