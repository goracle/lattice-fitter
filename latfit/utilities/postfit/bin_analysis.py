"""Analyze fit window bins"""
import sys
import re
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import gvar
import latfit.utilities.read_file as rf
from latfit.utilities.postfit.fitwin import pr_best_fitwin
from latfit.utilities.postfit.fitwin import replace_inf_fitwin, win_nan
from latfit.utilities.postfit.fitwin import max_tmax, LENMIN
from latfit.utilities.postfit.fitwin import generate_continuous_windows

# p1 32c
CBEST = [
    [['0.33019(35)', '-3.08(36)'], ['0.5341(14)', '-17.1(1.6)'], ['0.6614(76)', '-15(12)']],
    [['0.33035(34)', '-3.24(35)'], ['0.5341(11)', '-17.1(1.3)'], ['0.6641(58)', '-19.6(9.3)']],
    [['0.33035(34)', '-3.30(34)'], ['0.53320(81)', '-15.99(97)'], ['0.6648(32)', '-20.7(5.1)']],
    [['0.33035(34)', '-3.30(34)'], ['0.53352(57)', '-16.37(68)'], ['0.6641(21)', '-19.5(3.4)']],
    [['0.33055(33)', '-3.45(34)'], ['0.53327(46)', '-16.08(54)'], ['0.6661(15)', '-22.7(2.3)']],
    [['0.33053(33)', '-3.43(34)'], ['0.53349(45)', '-16.08(54)'], ['0.6654(13)', '-21.6(2.1)']],
    [['0.33055(32)', '-3.45(33)'], ['0.53338(36)', '-16.20(43)'], ['0.66540(78)', '-22.6(1.3)']],
    [['0.33055(32)', '-3.45(33)'], ['0.53327(32)', '-16.04(39)'], ['0.66618(60)', '-22.89(91)']],
    [['0.33055(32)', '-3.45(33)'], ['0.53327(32)', '-16.04(39)'], ['0.66618(60)', '-22.89(91)']]

]

# p11 32c (do not use)
CBEST = [
   # [['0.40458(35)', '-5.12(58)'], ['0.44803(26)', '-7.95(62)'], ['0.58217(50)', '-14.7(3.0)'], ['0.60224(60)', '-19.7(1.2)']]
    [['0.40458(35)', '-5.12(58)'], ['0.44789(24)', '-7.62(58)'], ['0.58218(46)', '-14.7(2.8)'], ['0.60187(55)', '-18.6(1.1)']]

]
IGNORABLE_WINDOWS = [(9, 13), (10, 14)]

# default
CBEST = []
IGNORABLE_WINDOWS = []

# p0 32c

CBEST = [
    [['0.21080(44)', '-0.26(12)'], ['0.45676(52)', '-13.18(60)'], ['0.6179(23)', '-25.0(2.4)'], ['0.7285(63)', '-34(11)']],
    [['0.21080(44)', '-0.30(12)'], ['0.45641(52)', '-13.18(60)'], ['0.6132(15)', '-20.1(1.5)'], ['0.7246(30)', '-26.8(5.7)']],
    [['0.21080(44)', '-0.30(12)'], ['0.45667(44)', '-13.08(51)'], ['0.61418(98)', '-21.5(1.0)'], ['0.7224(21)', '-23.8(4.1)']]

]
IGNORABLE_WINDOWS = []


# don't count these fit windows for continuity check
# they are likely overfit,
# (in the case of an overfit cut : a demand that chi^2/dof >= 1)
# or failing for another acceptable/known reason
# so the data will necessarily be missing


def augment_overfit(wins):
    """Augment likely overfit fit window
    list with all subset windows"""
    wins = set(wins)
    for win in sorted(list(wins)):
        tmin = win[0]
        tmax = win[1]
        for i in range(tmax-tmin-LENMIN):
            wins.add((tmin+i+1, tmax))
    for win in sorted(list(wins)):
        tmin = win[0]
        tmax = win[1]
        for i in range(tmax-tmin-LENMIN):
            wins.add((tmin, tmax-i-1))
    return sorted(list(wins))

# the assumption here is that all sub-windows in these windows
# have also been checked and are also (likely) overfit
IGNORABLE_WINDOWS = augment_overfit(IGNORABLE_WINDOWS)

print("CBEST =", CBEST)
print("IGNORABLE_WINDOWS =", IGNORABLE_WINDOWS)

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

ALLOW_ENERGY, ALLOW_PHASE = fill_best(CBEST)

def consistency(_):
    """dummy function, to be filled in later"""


try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

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
def continuous_tmax(tot_new):
    """Check continuous tmax"""
    maxtmax = max_tmax(tot_new)
    cwin = generate_continuous_windows(maxtmax)
    print("cwin=", cwin)
    for tmin in maxtmax:
        check_set = set()
        check_set = check_set.union(set(IGNORABLE_WINDOWS))
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
def plot_t_dep(tot, dim, item_num, title, units):
    """Plot the tmin dependence of an item"""
    print("plotting t dependence of dim", dim, "item:", title)
    tot_new = [i[dim][item_num] for i in tot if not np.isnan(
        gvar.gvar(i[dim][item_num][0]).val)]
    try:
        if not ALLOW_ENERGY and not ALLOW_PHASE:
            tot_new = check_fitwin_continuity(tot_new)
    except AssertionError:
        print("fit windows are not continuous for dim, item:", dim, title)
        raise
        #tot_new = []
    if list(tot_new):
        quick_compare(tot_new, prin=True)
        plot_t_dep_totnew(tot_new, dim, title, units)
    else:
        print("not enough consistent results for a complete set of plots")
        sys.exit(1)

@PROFILE
def check_fitwin_continuity(tot_new):
    """Check fit window continuity
    up to the minimal separation of tmin, tmax"""
    continuous_tmin_singleton(tot_new)
    flag = 1
    while flag:
        try:
            continuous_tmax(tot_new)
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
def plot_t_dep_totnew(tot_new, dim, title, units):
    """Plot something (not nothing)"""
    yarr = []
    yerr = []
    xticks_min = []
    xticks_max = []
    itemprev = None
    fitwinprev = None
    itmin = [gvar.gvar(np.nan, np.inf), []]
    for item, _, fitwin in tot_new:
        fitwin = fitwin[1]
        item = gvar.gvar(item)
        trfitwin = (fitwin[0], fitwin[1] + 1)
        if item == itemprev and trfitwin == fitwinprev and len(
                tot_new) > 10:
            # decreasing tmax while holding tmin fixed will usually
            # not change the min, so don't plot these
            fitwinprev = fitwin
            itemprev = item
            print("omitting (item, dim, val(err), fitwindow):",
                  title, dim, item, fitwin)
            continue
        if np.isnan(item.val):
            continue
        yarr.append(item.val)
        yerr.append(item.sdev)
        if item.sdev <= itmin[0].sdev:
            itmin = (item, fitwin)
        xticks_min.append(str(fitwin[0]))
        xticks_max.append(str(fitwin[1]))
        fitwinprev = fitwin
        itemprev = item
    if itmin[1] not in plot_t_dep.fitwin_votes:
        plot_t_dep.fitwin_votes[itmin[1]] = 0
    plot_t_dep.fitwin_votes[itmin[1]] += 1
    xarr = list(range(len(xticks_min)))
    assert len(xticks_min) == len(xticks_max)

    fname = sys.argv[1]

    try:
        tmin = int(fname.split('tmin')[1].split('.')[0])
    except IndexError:
        tmin = rf.earliest_time(fname)

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
        print("minimum error:", itmin[0], "fit window:", itmin[1])
        app_itmin(itmin[0])
        print("saving fig:", save_str)
        pdf.savefig()
    plt.show()
plot_t_dep.fitwin_votes = {}
plot_t_dep.coll = []

@PROFILE
def print_tot(tot):
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
    for dim, _ in enumerate(tot[0]):
        plot_t_dep(tot, dim, 0, 'Energy', 'lattice units')
        plot_t_dep(tot, dim, 1, 'Phase shift', 'degrees')
    print(plot_t_dep.coll)
    pr_best_fitwin(plot_t_dep.fitwin_votes)
    prune_cbest()

@PROFILE
def tot_to_stat(res, sys_err):
    """Get result object which has separate stat
    and sys errors"""
    err = res.sdev
    assert err > sys_err, (err, sys_err)
    err = np.sqrt(err**2-sys_err**2)
    ret = gvar.gvar(res.val, err)
    return ret

def process_res_to_best(min_en, min_ph):
    """Process results to append to 'best'
    known list"""
    min_enb = [i for i, _, _ in min_en]
    min_phb = [i for i, _, _ in min_ph]
    min_resb = [
        list(i) for i in zip(min_enb, min_phb) if list(i)]
    update_best(min_resb)


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

def prune_cbest(cbest=None):
    """Prune the best known so only
    the smallest error params remain.  Print the result."""
    cbest = update_best.cbest if cbest is None else cbest
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

def update_best(new_best):
    """Update best lists"""
    cbest = update_best.cbest
    if cbest:
        if compare_bests(new_best, update_best.cbest):
            print("adding new best known params:", new_best)
            new_best = match_arrs(update_best.cbest[0], new_best)
            update_best.cbest.append(new_best)
            prune_cbest(cbest)
            cbest = [update_best.cbest[0]]
            print("current best known params list:", cbest)
    find_best.allow_energy, find_best.allow_phase = fill_best(cbest)
update_best.cbest = CBEST

def find_best():
    """Find best known result"""
    assert find_best.sel is not None, find_best.sel
    sel = find_best.sel
    if sel == 'phase':
        best = find_best.allow_phase
    if sel == 'energy':
        best = find_best.allow_energy
    return best
find_best.sel = None
find_best.allow_phase = ALLOW_PHASE
find_best.allow_energy = ALLOW_ENERGY


@PROFILE
def app_itmin(itmin):
    """Append next minimum result"""
    app_itmin.curr.append(str(itmin))
    if len(app_itmin.curr) == 2:
        plot_t_dep.coll.append(app_itmin.curr)
        app_itmin.curr = []
app_itmin.curr = []

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
def select_ph_en(sel):
    """Select which quantity we will analyze now"""
    assert sel in ('phase', 'energy'), sel
    find_best.sel = sel

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

@PROFILE
def quick_compare(tot_new, prin=False):
    """Check final results for consistency"""
    for item, _, fitwin in tot_new:
        item = gvar.gvar(item)
        for item2, _, fitwin2 in tot_new:
            item2 = gvar.gvar(item2)
            assert consistency(item, item2, prin=prin), (
                item, item2, fitwin, fitwin2)
