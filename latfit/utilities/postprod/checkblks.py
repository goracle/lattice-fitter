"""Checksums part of h5jack"""
import sys
import re
import itertools
import pickle
import numpy as np

from latfit.utilities import read_file as rf
from latfit.utilities import write_discon as wd
from latfit.utilities import op_compose as opc

# settings for checks
# diagram to look at for bubble subtraction test
# TESTKEY = 'FigureBub2_mom000'
TESTKEY = 'FigureV_sep3_mom1src000_mom2src000_mom1snk000'
TESTKEY = 'FigureD_vec_sep4_mom1src00_1_mom2src001_mom1snk001'
TESTKEY = 'FigureCv3R_sep3_momsrc000_momsnk000'
TESTKEY = 'FigureHbub_scalar_mom000'
TESTKEY = ''


# (0..N or ALL for all time slices) for a diagram TESTKEY2
TESTKEY2 = 'FigureV_sep4_mom1src000_mom2src000_mom1snk000'
TESTKEY2 = ''
# Print out the jackknife block at t=TSLICE
TSLICE = 0

# run a test on a 4^4 latice
TEST44 = True
TEST44 = False

# run a test on a 24c x 64 lattice
TEST24C = True
TEST24C = False
TEST44 = True if TEST24C else TEST44

# free field check
FREEFIELD = True
FREEFIELD = False


#### DO NOT MODIFY THE BELOW

def fold_time(*xargs, **kwargs):
    """dummy function"""
    assert None
    if xargs or kwargs:
        pass
    return xargs

# dynamically set by h5jack
LT = np.nan
SKIP_VEC = None
EXACT_CONFIGS = []
AVGTSRC = None
DOAMA = None
SKIP_VEC = None
WRITE_INDIVIDUAL = None
ENSEMBLE_DICT = {}

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

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

def convert_traj(traj):
    """convert a string like '1540_exact'
    to int, e.g. 1540
    """
    traj = str(traj)
    return int(re.search(r"(\d)+", traj).group(0))

def getopstr(opstr):
    """I0/pipisigma_A_1PLUS_mom000->
    A_1PLUS_mom000
    """
    split = opstr.split('_')[1:]
    for i, sub in enumerate(split):
        if 'pipi' in sub or 'sigma' in sub or 'rho' in sub:
            continue
        split = split[i:]
        break
    ret = ''
    for sub in split:
        ret = ret+sub+'_'
    ret = ret[:-1]
    return ret

def strip_op(op1):
    """Strips off extra specifiers including polarizations."""
    return op1.split('?')[0]

def check_key(key):
    """Only look at the key in question, tell parent to skip the rest"""
    if (TESTKEY and key != TESTKEY) or (TESTKEY2 and key != TESTKEY2):
        retval = False
    else:
        retval = True
    return retval

@PROFILE
def check_dup_configs(blks):
    """Check for the same config being written twice for some reason"""
    if hasattr(blks, '__iter__'):
        for i in blks:
            try:
                try_check_dup(i, blks)
            except AssertionError:
                print("config index 0 same as config",
                      "index 1 for block with name:")
                if isinstance(blks, dict):
                    print(i)
                else:
                    print(None)
                sys.exit(1)
    else:
        check_dup_single(blks, i)

def try_check_dup(iinblks, blks):
    """Try block in check_dup_configs
    """
    if isinstance(blks, dict):
        if not TEST44:
            if len(blks[iinblks].shape) == 2:
                assert blks[iinblks][0, 0] != blks[iinblks][1, 0]
            else:
                assert blks[iinblks][0, 0, 0] != blks[iinblks][1, 0, 0]
    else:
        if len(iinblks.shape) == 2:
            assert iinblks[0, 0] != iinblks[1, 0]
        else:
            assert iinblks[0, 0, 0] != iinblks[1, 0, 0]

def check_dup_single(blk, idx):
    """Check dup configs for a single block"""
    if blk is not None:
        try:
            if len(blk.shape) == 2:
                assert blk[0, 0] != blk[1, 0]
            else:
                assert blk[0, 0, 0] != blk[1, 0, 0]
        except AssertionError:
            print("config index 0 same as config",
                  "index 1 for block with name:")
            print(idx)
            sys.exit(1)

@PROFILE
def debugprint(blk, base, pstr=None):
    """Print some debug info about this jackknife block"""
    temp = fold_time(blk, base)
    if pstr is None:
        print(base)
        print(blk)
    else:
        print(pstr, blk)
    print(blk.shape)
    printt = True
    for i, row in enumerate(temp):
        print('traj=', i)
        for j in row:
            print(j)
    return printt

@PROFILE
def check_ids(ensemble):
    """Check the ensemble id file to be sure
    not to run processing parameters from a different ensemble"""
    assert ENSEMBLE_DICT, "ensemble params not set"
    ens = ENSEMBLE_DICT[ensemble]
    tsep = ens['tsep']
    tdis_max = ens['tdis_max']
    tstep = ens['tstep']
    ids_check = [tsep, tdis_max, tstep, DOAMA, SKIP_VEC]
    ids_check = np.asarray(ids_check)
    ids = pickle.load(open('ids.p', "rb"))
    assert np.all(ids == ids_check),\
        "wrong ensemble. [TSEP, TDIS_MAX, TSTEP,  DOAMA, SKIP_VEC]"+\
        " should be:"+str(ids)+" instead of:"+str(ids_check)+" ens:"+str(
            ensemble)
    return ids



@PROFILE
def check_inner_outer(ocs, allkeys, auxkeys):
    """Check to make sure the inner pion has energy >= outer pion
    """
    allkeys = set(allkeys)
    auxkeys = set(auxkeys)
    for opa in ocs:
        for diag, _, _ in ocs[opa]:
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
            for diag, _, _ in ocs[opa]:
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
def discount_useless(useless, count):
    """Correct count of useless diagrams"""
    try:
        # we don't analyze all of pcom for I=1 yet
        if not (rf.vecp(useless) and rf.norm2(wd.momtotal(
                rf.mom(useless)))):
            if not rf.checkp(useless) and not (rf.vecp(
                    useless) and rf.allzero(
                        rf.mom(useless))):
                print("unused diagram:", useless)
                count -= 1
    except TypeError:
        print(useless)
        sys.exit(1)
    return count

def checksum_diagrams(ocs, allblks, auxblks):
    """do a checksum to make sure we have all the diagrams we need"""
    for i in ocs:
        print(i)
    check_count_of_diagrams(ocs, "I0")
    check_count_of_diagrams(ocs, "I2")
    if not SKIP_VEC:
        check_count_of_diagrams(ocs, "I1")
    check_match_oplist(ocs)
    check_inner_outer(
        ocs, allblks.keys() | set(), auxblks.keys() | set())
    unused = set()
    # 'FigureV', 'FigureCv3', 'FigureCv3R',
    for fig in ['FigureR', 'FigureC', 'FigureD',
                'FigureBub2', 'FigureT']:
        unused = find_unused(
            ocs, allblks.keys() | set(),
            auxblks.keys() | set(), fig=fig).union(unused)
    count = len(unused)
    for useless in sorted(list(unused)):
        count = discount_useless(useless, count)
    print("length of unused=", len(unused))
    assert unused or count == len(unused), \
        "Unused (non-vec) diagrams exist."

def check_count_of_diagrams(ocs, isospin_str='I0'):
    """count how many diagrams go into irrep/iso projection
    if it does not match the expected, abort
    """
    checks = opc.generate_checksums(isospin_str[-1])
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
            lopa = len(ocs[opa])
            polcount = int(ocs[opa][0][2])
            if not irrop == getopstr(opa) or isospin_str not in opa:
                continue
            print("isospin checksumming op=", opa, "for correctness")

            # get isocount
            if 'rho' in opa:
                isocount = 1 if not polcount else polcount
            elif 'sigma' in opa:
                isocount = 2
            else:
                assert 'pipi' in opa, "bad operator:"+str(ocs[opa])
                isocount = 4 if 'I0' in opa else 2
            # check this!
            assert lopa % isocount == 0,\
                "bad isospin projection count."+str(ocs[opa])+\
                " in "+str(opa)

            # print checksum
            print("isocount =", isocount, "number of diagrams in op=", lopa)
            print("divided by isocount=", lopa/isocount)

            # store checksum
            counter_checks[irrop] += lopa/isocount

        assert counter_checks[irrop] == checks[irrop], "bad checksum, "+\
            " number of expected diagrams"+\
            " does not match:"+str(checks[irrop])+" vs. "+str(
                counter_checks[irrop])+" "+str(irrop)

@PROFILE
def check_match_oplist(ocs):
    """A more stringent check:
    Generate the momentum combinations for an operator
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
            for diag, _, _ in ocs[opa]:
                figlist.add(rf.mom_prefix(diag))
            for diag, _ in opcheck[irrop]:
                momlist.add(rf.mom_suffix(diag))
            for prefix, suffix in itertools.product(figlist, momlist):
                checkstr = prefix + '_' + suffix
                assert checkstr in diaglist,\
                    "Missing momentum combination:"+\
                    str(checkstr)+" in op="+str(opa)+"="+str(ocs[opa])+\
                    " figlist="+\
                    str(figlist)+" momlist="+str(momlist)

@PROFILE
def check_ama(blknametocheck, sloppyblks, exactblks, sloppysubtractionblks):
    """Check block for consistency across sloppy and exact samples
    return the consistency-checked block lengths
    """
    blk = blknametocheck
    if AVGTSRC:
        len_exact, lte = exactblks[blk].shape
        len_exact_check, lte_check = sloppysubtractionblks[blk].shape
        len_sloppy, lts = sloppyblks[blk].shape
    else:
        len_exact, lte, _ = exactblks[blk].shape
        len_exact_check, lte_check, _ = sloppysubtractionblks[blk].shape
        len_sloppy, lts, _ = sloppyblks[blk].shape

    # do some checks
    assert len_exact_check == len_exact, "subtraction term in correction \
    has unequal config amount to exact:"+str(
        len_exact)+", "+str(len_exact_check)
    nexact = len(EXACT_CONFIGS)
    assert len_exact_check == nexact, "subtraction term in correction \
    has unequal config amount to global exact:"+str(
        nexact)+", "+str(len_exact_check)
    assert lte == lte_check, "subtraction term in correction\
    has unequal time extent to exact:"+str(lte)+", "+str(lte_check)
    assert lte == LT, "exact configs have unequal time extent \
    to global time:"+str(lte)+", "+str(LT)
    assert lts == LT, "sloppy configs have unequal time extent \
    to global time:"+str(lts)+", "+str(LT)

    print("Creating super-jackknife block=", blk,
          "with", nexact, "exact configs and", len_sloppy, "sloppy configs")
    return len_sloppy, len_exact
