"""Makes statements (and assertions)"""
from latfit.utilities import read_file as rf
import latfit.fit_funcs

def gevp_statements(gevp_dirs, gevp, dim, lt_vec, add_const_vec):
    """Make gevp related statements"""
    print("GEVP directories:", gevp_dirs)
    #GEVP_DIRS = np.delete(GEVP_DIRS, 1, axis=1)
    #GEVP_DIRS = np.delete(GEVP_DIRS, 1, axis=0)
    mult = len(GEVP_DIRS) if GEVP else 1
    if gevp:
        assert dim == mult, "Error in GEVP_DIRS length."
        #assert not(LOG and PIONRATIO), \
        #    "Taking a log is improper when doing a pion ratio fit."
    assert len(lt_vec) == mult, "Must set time separation separately for"+\
    " each diagonal element of GEVP matrix"
    assert len(add_const_vec) == mult, "Must separately set, whether or"+\
        " not to use an additive constant in the fit function,"+\
        " for each diagonal element of GEVP matrix"
    return mult

def start_params_pencils(start_params, origl, num_pencils,
                           mult, sys_energy_guess):
    """start parameter statements"""
    if len(start_params) % 2 == 1 and origl > 1:
        start_params = list(start_params[:-1])*mult
        start_params.append(sys_energy_guess)
        start_params = start_params*2**num_pencils
    else:
        start_params = (list(start_params)*mult)*2**num_pencils
    return start_params

def fit_func_statements(use_fixed_mass, update, tstep, fits):
    """Fit function statements"""
    latfit.fit_funcs.USE_FIXED_MASS = use_fixed_mass
    update.tstep = tstep # revert back
    fits.select(update)

def rescale_and_statements(eff_mass, eff_mass_method, rescale,
                           delta_e_around_the_world,
                           delta_e2_around_the_world):
    """effective mass statements"""
    if eff_mass:
        if eff_mass_method in [1, 3, 4]:
            print("rescale set to 1.0")
            rescale = 1.0
    print("Assuming slowest around the world term particle is stationary.")
    print("delta E around the world (first term)=", delta_e_around_the_world)
    print("2nd order around the world term, delta E=",
          delta_e2_around_the_world)

def asserts_one(eff_mass_method, matrix_subtraction,
                jackknife_fit, num_pencils, jackknife):
    """some asserts"""
    assert eff_mass_method == 4 or not matrix_subtraction, "Matrix"+\
        " subtraction supported"+\
        " only with eff mass method 4"
    assert jackknife_fit == 'DOUBLE', "Other jackknife fitting"+\
        " methods no longer supported."
    assert num_pencils == 0, "this feature is less tested, "+\
        " use at your own risk (safest to have NUM_PENCILS==0)"
    assert jackknife == 'YES', "no jackknife correction if not YES"

def asserts_two(irrep, fit_spacing_correction, isospin, pionratio):
    """some asserts (2/?)"""
    assert 'avg' in irrep or 'mom111' not in irrep or 'A' not in irrep, \
        "A1_avg_mom111 is the "+\
        "averaged over rows, A1_mom111 is one row.  "+\
        "(Comment out if one row is what was intended).  IRREP="+str(irrep)
    assert not fit_spacing_correction or isospin == 2 or pionratio, \
        "isospin 2 is the only user of this"+\
        " lattice spacing correction method"

def asserts_three(momstr, delta_e_around_the_world,
                  delta_e2_around_the_world, gevp, gevp_deriv):
    """some asserts"""
    if rf.norm2(rf.procmom(momstr)) == 0:
        assert np.all(delta_e_around_the_world == 0.0), \
            "only 1 constant in COMP frame:"+str(delta_e_around_the_world)
        assert delta_e2_around_the_world is None, \
            "only 1 constant in COMP frame"
    if gevp:
        print("GEVP derivative being taken:", gevp_deriv)

def bin_time_statements(binnum, use_late_times, t0f, biased_speedup):
    """more statements"""
    print("Binning configs.  Bin size =", binnum)
    assert not use_late_times, "method is based on flawed assumptions."
    assert tof != "ROUND", "bad systematic errors result from this option"
    assert not biased_speedup, "it is biased.  do not use."
    assert tof != 'ROUND',\
        "too much systematic error if t-t0!=const." # ceil(t/2)
    assert tof != 'LOOP',\
        "too much systematic error if t-t0!=const." # ceil(t/2)
    assert 'TMINUS' in tof,\
        "t-t0=const. for best known systematic error bound."

def bin_statements(binnum, elim_jkconf_list, half,
                   only_small_fit_ranges, range_length_min):
    """some statements"""
    assert binnum == 1 or not elim_jkconf_list, "not supported"
    assert not elim_jkconf_list or half == "full", "not supported"
    # we can't fit to 0 length subsets
    assert not only_small_fit_ranges or range_length_min

def delta_e2_mod(systematic_est, matrix_subtraction, pionratio,
                 delta_e2_around_the_world, delta_e_around_the_world):
    """some more statements"""
    assert not systematic_est, "cruft; should be removed eventually"
    #assert not PIONRATIO or ISOSPIN == 2
    #assert MATRIX_SUBTRACTION or not PIONRATIO
    if delta_e2_around_the_world is not None:
        delta_e2_around_the_world -= delta_e_around_the_world
    if PIONRATIO:
        #assert ISOSPIN == 2
        print("using pion ratio method, PIONRATIO:", PIONRATIO)
    else:
        print("not using pion ratio method, PIONRATIO:", PIONRATIO)
    return delta_e2_around_the_world

def matsub_statements(matrix_subtraction, irrep, isospin, gevp, noatwsub):
    """some statements"""
    assert not matrix_subtraction or '000' in irrep or isospin == 0 or\
        isospin == 1, str(irrep)
    if isospin == 2 and gevp:
        assert not noatwsub
        assert matrix_subtraction or irrep != 'A_1PLUS_mom000'
    if not noatwsub:
        print("matrix subtraction:", matrix_subtraction)
    assert isospin != 1 or noatwsub, "I=1 has no ATW terms."

def superjackknife_statements(check_ids_minus_2, superjack_cutoff):
    """superjackknife statements"""
    #if check_ids()[-2]:
    if check_ids_minus_2:
        assert superjack_cutoff,\
            "AMA is turned on.  super jackknife cutoff should be non-zero"
    else:
        assert not superjack_cutoff,\
            "AMA is turned off.  super jackknife cutoff should be zero"
