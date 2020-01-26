"""Analyze fit window bins"""
import sys
import re
import pickle
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import gvar
from latfit.utilities.postfit.fitwin import pr_best_fitwin
from latfit.utilities.postfit.fitwin import replace_inf_fitwin, win_nan
from latfit.utilities.postfit.fitwin import max_tmax
from latfit.utilities.postfit.fitwin import generate_continuous_windows
from latfit.utilities.postfit.cuts import consistency
from latfit.utilities.postfit.strproc import tmin_param, min_fit_file
from latfit.utilities.combine_pickle import main as getdts
from latfit.config import ISOSPIN, LATTICE_ENSEMBLE, IRREP

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

def round_gvar(item):
    """do not store extra error digits in a gvar item"""
    item = gvar.gvar(item)
    item = str(item)
    item = gvar.gvar(item)
    return item

def fill_best(cbest):
    """Fill the ALLOW buffers with current best known result"""
    rete = []
    retph = []
    for i in cbest:
        aph = []
        aen = []
        for j in i:
            energy = round_gvar(j[0])
            phase = round_gvar(j[1])
            aen.append(energy)
            aph.append(phase)
        rete.append(aen)
        retph.append(aph)
    return rete, retph

@PROFILE
def cut_tmin(tot_new, tocut):
    """Cut tmin"""
    todel = []
    for i, (_, _, fitwin) in enumerate(tot_new):
        fitwin = fitwin[1] # cut out the fit range info
        assert isinstance(fitwin[0], np.float), fitwin
        if fitwin[0] in tocut or fitwin[0] < min(tocut):
            todel.append(i)
    ret = np.delete(tot_new, todel, axis=0)
    return ret

@PROFILE
def continuous_tmax(tot_new, ignorable_windows):
    """Check continuous tmax"""
    maxtmax = max_tmax(tot_new)
    cwin = generate_continuous_windows(maxtmax)
    print("cwin=", cwin)
    for tmin in maxtmax:
        check_set = set()
        check_set = check_set.union(set(ignorable_windows))
        for _, _, fitwin in tot_new:
            fitwin = fitwin[1]
            if fitwin[0] == tmin:
                check_set.add(fitwin)
        #print("starting fit windows("+str(tmin)+"):",
        #      check_set, "vs.", cwin[tmin])

        # check tmax is continuous
        assert not cwin[tmin]-check_set, (cwin[tmin]-check_set)
            #(cwin[tmin], check_set, cwin[tmin]-check_set)
        # sanity check
        #assert not check_set-cwin[tmin],\
            #(cwin[tmin], check_set, check_set-cwin[tmin])

@PROFILE
def continuous_tmin_singleton(tot_new):
    """Check for continous tmin, singleton cut"""

    # singleton cut
    assert len(tot_new) > 1, tot_new
    maxtmax = max_tmax(tot_new)
    tmin_cont = np.arange(min(maxtmax), max(maxtmax)+1)

    # check tmin is continuous
    assert not set(tmin_cont)-set(maxtmax), list(maxtmax)


@PROFILE
def drop_extra_info(ilist):
    """Drop fit range data from a 'tot' list item
    (useful for prints)
    i[0][0]==energy
    i[1][0]==phase
    (in general i[j][0] == result)
    i[j][1] == sys_err
    i[j][2] == (fit_range, fit window)
    """
    # find the common fit range, for this fit window if it exists
    fit_range = None
    for i in ilist:
        fitr1 = i[0][2][0]
        fitr2 = i[1][2][0]
        if isinstance(fitr1, int) or isinstance(fitr2, int):
            fit_range = None
            break
        assert hasattr(fitr2, '__iter__'), (
            fitr1, fitr2, i)
        if fit_range_equality(fitr1, fitr2):
            if fit_range is None:
                fit_range = fitr1
            else:
                if not fit_range_equality(fitr1, fit_range):
                    fit_range = None
                    break
        else:
            fit_range = None
            break
    # printable results
    ret = []
    if fit_range is None:
        # ret is (energy, fit range), (phase, fit range)
        for i in ilist:
            if 'nan' in str(i[0][0]):
                toapp = []
            else:
                fitr1 = i[0][2][0]
                fitr2 = i[1][2][0]
                if isinstance(fitr1, int):
                    assert not fitr1
                    fitr1 = fitr2
                if isinstance(fitr2, int):
                    assert not fitr2
                    fitr2 = fitr1
                    if isinstance(fitr1, int):
                        continue
                assert hasattr(fitr2, '__iter__'), (
                    fitr1, fitr2, i)
                # check for sub consistency of fit ranges
                if fit_range_equality(fitr1, fitr2):
                    toapp = ([i[0][0], i[1][0]], fitr1)
                else:
                    #assert None, (fitr1, fitr2)
                    toapp = [(i[0][0], fitr1), (i[1][0], fitr2)]
            ret.append(toapp)
    else:
        # ret is ((energy, phase)..., fit range)
        ret = [[i[0][0], i[1][0]] if 'nan' not in str(
            i[0][0]) else [] for i in ilist]

    # get common fit window, make sure there is a common fit window
    fitwin = [[i[0][2][1], i[1][2][1]] if 'nan' not in str(
        i[0][0]) else [] for i in ilist]
    fitr = [[i[0][2][0], i[1][2][0]] if 'nan' not in str(
        i[0][0]) else [] for i in ilist]
    fitw = None
    hatt = False
    for i, j in zip(fitwin, fitr):
        i = list(i)
        i = replace_inf_fitwin(i)
        j = list(j)
        if not i and not j:
            continue
        assert i or j, (i, j)
        if fitw is None:
            fitw = i
            hatt = hasattr(fitw[0], '__iter__')
        else:
            if hatt:
                con1 = list(fitw[0]) == list(i[0])
                con2 = list(fitw[1]) == list(i[1])
                con = con1 or con2
            else:
                con = list(fitw) == list(i)
            assert con, (fitw, i, j)
    if hatt:
        con1 = list(fitw[0]) == list(fitw[1])
        con2 = np.inf == fitw[0][1] or win_nan(fitw[0])
        con3 = np.inf == fitw[1][1] or win_nan(fitw[1])
        assert con1 or con2 or con3, fitw
        if con1:
            fitw = fitw[0]
        elif con2:
            fitw = fitw[1]
        elif con3:
            fitw = fitw[0]
    return (fitw, fit_range, ret)

@PROFILE
def plot_t_dep(tot, plot_info, fitwin_votes, toapp, best_info):
    """Plot the tmin dependence of an item"""
    dim, item_num, title, units, fname = plot_info
    cond, ignorable_windows = best_info
    print("plotting t dependence of dim", dim, "item:", title)
    tot_new = [i[dim][item_num] for i in tot if not np.isnan(
        gvar.gvar(i[dim][item_num][0]).val)]
    try:
        if cond:
            tot_new = check_fitwin_continuity(tot_new, ignorable_windows)
    except AssertionError:
        print("fit windows are not continuous for dim, item:", dim, title)
        raise
        #tot_new = []
    if list(tot_new):
        quick_compare(tot_new, prin=True)
        plot_info = dim, title, units, fname
        fitwin_votes, toapp = plot_t_dep_totnew(tot_new, plot_info, fitwin_votes, toapp)
    else:
        print("not enough consistent results for a complete set of plots")
        sys.exit(1)
    return fitwin_votes, toapp

@PROFILE
def check_fitwin_continuity(tot_new, ignorable_windows):
    """Check fit window continuity
    up to the minimal separation of tmin, tmax"""
    continuous_tmin_singleton(tot_new)
    flag = 1
    while flag:
        try:
            continuous_tmax(tot_new, ignorable_windows)
            flag = 0
        except AssertionError as err:
            tocut = set()
            err = ast.literal_eval(str(err))
            for i in err:
                assert len(i) == 2, (i, err)
                tocut.add(i[0])
            print("cutting:", tocut)
            tot_new = cut_tmin(tot_new, tocut)
            continuous_tmin_singleton(tot_new)
    return tot_new

@PROFILE
def plot_t_dep_totnew(tot_new, plot_info, fitwin_votes, toapp):
    """Plot something (not nothing)"""
    dim, title, units, fname = plot_info
    yarr = []
    yerr = []
    xticks_min = []
    xticks_max = []
    itemprev = None
    fitwinprev = None
    itmin = [gvar.gvar(np.nan, np.inf), []]
    for item, _, fitwin in tot_new:
        fitrange, fitwindow = fitwin
        item = gvar.gvar(item)
        trfitwin = (fitwindow[0] + 1, fitwindow[1])
        if item == itemprev and trfitwin == fitwinprev and len(
                tot_new) > 10:
            # decreasing tmax while holding tmin fixed
            # will usually not change the min,
            # so don't plot these
            fitwinprev = fitwindow
            itemprev = item
            print("omitting (item, dim, val(err), fitwindow):",
                  title, dim, item, fitwindow)
            continue
        if np.isnan(item.val):
            continue
        yarr.append(item.val)
        yerr.append(item.sdev)
        if item.sdev < itmin[0].sdev:
            itmin = (item, fitrange, fitwindow)
        xticks_min.append(str(fitwindow[0]))
        xticks_max.append(str(fitwindow[1]))
        fitwinprev = fitwindow
        itemprev = item
    if itmin[2] not in fitwin_votes:
        fitwin_votes[itmin[2]] = 0
    fitwin_votes[itmin[2]] += 1
    xarr = list(range(len(xticks_min)))
    assert len(xticks_min) == len(xticks_max)

    fname = sys.argv[1]

    tmin = tmin_param(fname)

    # print info related to final fits
    to_include(itmin, dim, title)

    save_str = re.sub('phase_shift_', '', fname)
    save_str = re.sub('energy_', '', save_str)
    save_str = re.sub(' ', '_', save_str)
    save_str = re.sub(r'.p$', '_tdep_'+title+'_state'+str(
        dim)+'.pdf', save_str)

    with PdfPages(save_str) as pdf:
        ax1 = plt.subplot(1, 1, 1)
        ax1.errorbar(xarr, yarr, yerr=yerr, linestyle="None")
        ax1.set_xticks(xarr)
        ax1.set_xticklabels(xticks_min)
        ax1.set_xlabel('fit window tmin (lattice units)')
        ax1.set_ylabel(title+' ('+units+')')

        ax2 = ax1.twiny()
        ax2.set_xticks(xarr)
        ax2.set_xticklabels(xticks_max)
        ax2.set_xlabel('fit window tmax (lattice units)')
        ax2.set_xlim(ax1.get_xlim())

        # ok
        #ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
        #ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
        #ax2.spines['bottom'].set_position(('outward', 36))

        ax2.set_title(title+' vs. '+'fit window; state '+str(
            dim)+","+r' $t_{min,param}=$'+str(tmin), y=1.12)
        plt.subplots_adjust(top=0.85)
        print("minimum error:", itmin[0], "fit range:", itmin[1], "fit window:", itmin[2])
        toapp.append(str(itmin[0]))
        print("saving fig:", save_str)
        pdf.savefig()
    plt.show()
    return fitwin_votes, toapp

def to_include(itmin, dim, title):
    """Show the include.py settings just learned"""
    sel = [[j for j in i] for i in itmin[1]]
    fitwin = itmin[2]
    rest = title.lower()
    tosave = [sel, rest, dim, IRREP,
              LATTICE_ENSEMBLE, ISOSPIN, fitwin]
    print("INCLUDE =", sel)
    print("PARAM_OF_INTEREST = '"+rest+"'")
    print("DIMSELECT =", dim)
    print("IRREP =", IRREP)
    print("LATTICE_ENSEMBLE =", LATTICE_ENSEMBLE)
    print("ISOSPIN =", ISOSPIN)
    print("FIT_EXCL = invinc(INCLUDE,", str(fitwin)+")")
    tminus, dt2 = getdts(sel)
    tosave.append(tminus)
    tosave.append(dt2)
    print("T0 =", tminus)
    if dt2 is not None:
        print("DELTA_T_MATRIX_SUBTRACTION =", dt2)
    saven = min_fit_file(dim, rest)
    fn1 = open(saven+'.p', 'wb')
    pickle.dump(tosave, fn1)

@PROFILE
def print_tot(fname, tot, cbest, ignorable_windows):
    """Print results vs. tmin"""
    tot_new = []
    lenmax = 0
    print("summary results:")
    for i in tot:
        i = list(i)
        lenmax = max(lenmax, len(i))
        if i and len(i) == lenmax:
            tot_new.append(i)
        if i:
            topr = drop_extra_info(i)
            if topr:
                fitwin, fit_range, res = topr
                print('fitwin =', fitwin)
                if fit_range is not None:
                    print("fit range:", fit_range)
                for idx, item in enumerate(res):
                    print("dim", idx, ":", item)
    tot = tot_new
    fitwin_votes = {}
    coll = []
    toapp = []
    for dim, _ in enumerate(tot[0]):
        best_info = (not cbest, ignorable_windows)
        plot_info = (dim, 0, 'Energy', 'lattice units', fname)
        fitwin_votes, toapp = plot_t_dep(tot, plot_info, fitwin_votes, toapp, best_info)
        plot_info = (dim, 1, 'Phase Shift', 'degrees', fname)
        fitwin_votes, toapp = plot_t_dep(tot, plot_info, fitwin_votes, toapp, best_info)
        coll.append(toapp)
        toapp = []
    print(coll)
    pr_best_fitwin(fitwin_votes)
    cbest.append(coll)
    prune_cbest(cbest)
    return cbest

def compare_bests(new, curr):
    """Compare new best to current best
    to see if an update is needed"""
    ret = False
    for ibest in curr:
        new = match_arrs(ibest, new)
        assert len(new) == len(ibest), (new, ibest)
        for jit, kit in zip(ibest, new):
            if 'inf' in str(kit) or 'inf' in str(jit):
                continue
            if 'nan' in str(kit) or 'nan' in str(jit):
                continue
            enerr = gvar.gvar(jit[0]).sdev
            pherr = gvar.gvar(jit[1]).sdev
            enerr_new = gvar.gvar(kit[0]).sdev
            pherr_new = gvar.gvar(kit[1]).sdev
            if enerr > enerr_new or pherr > pherr_new:
                ret = True
                break
    return ret

def prune_cbest(cbest):
    """Prune the best known so only
    the smallest error params remain.  Print the result."""
    if len(cbest) == 1:
        cnew = cbest[0]
    elif not cbest:
        cnew = cbest
    else:
        # diml = len(cbest[0])
        cnew = None
        for ibest in cbest:
            if cnew is None:
                cnew = ibest
                continue
            assert len(cnew) == len(ibest)
            for idx, (jit, kit) in enumerate(zip(ibest, cnew)):
                enerr = gvar.gvar(jit[0]).sdev
                pherr = gvar.gvar(jit[1]).sdev
                enerr_new = gvar.gvar(kit[0]).sdev
                pherr_new = gvar.gvar(kit[1]).sdev
                if enerr < enerr_new:
                    cnew[idx][0] = str(jit[0])
                if pherr < pherr_new:
                    cnew[idx][1] = str(jit[1])
    print("pruned cbest list:", cnew)
    return cnew


def match_arrs(arr, new):
    """Match array lengths by extending the new array length"""
    arr = list(arr)
    new = list(new)
    dlen = len(arr) - len(new)
    assert dlen >= 0, (arr, new, dlen)
    if dlen:
        ext = arr[-1*dlen:]
        new.extend(ext)
    return new

@PROFILE
def fit_range_equality(fitr1, fitr2):
    """Check if two fit ranges are the same"""
    ret = True
    assert hasattr(fitr1, '__iter__'), (fitr1, fitr2)
    assert hasattr(fitr2, '__iter__'), (fitr1, fitr2)
    if fitr1 != [[]] and fitr2 != [[]]:
        assert len(fitr1) == len(fitr2), (fitr1, fitr2)
        for i, j in zip(fitr1, fitr2):
            if list(i) != list(j):
                ret = False
    else:
        ret = fitr1 == [[]] and fitr2 == [[]]
    return ret

def consis_tot(tot):
    """Check tot for consistency"""
    tot = np.array(tot)
    for opa in range(tot.shape[1]):
        quick_compare(tot[:, opa, 0], prin=False)
        quick_compare(tot[:, opa, 1], prin=False)
    tot = list(tot)

class BinInconsistency(Exception):
    """Error if bins are inconsistent"""
    @PROFILE
    def __init__(self, message='', prin=True):
        if prin:
            print("***ERROR***")
            print("bins give inconsistent results")
        super(BinInconsistency, self).__init__(message)
        self.message = message

@PROFILE
def quick_compare(tot_new, prin=False):
    """Check final results for consistency"""
    for item, _, fitwin in tot_new:
        item = gvar.gvar(item)
        for item2, _, fitwin2 in tot_new:
            item2 = gvar.gvar(item2)
            if not consistency(item, item2, prin=prin):
                print(item, item2, fitwin, fitwin2)
                raise BinInconsistency
