"""Get the actual fit function
fit function/processing function selection
"""
import sys
from copy import copy
from math import exp
import numpy as np
import latfit.analysis.misc as misc

# select which of the above library functions to use


def check_start_params_len(eff_mass, eff_mass_method, origl,
                           matrix_subtraction, delta_e2_around_the_world):
    """Perform a check on the length of the start parameters"""
    if eff_mass:

        # check len of start params
        #if origl != 1 and EFF_MASS_METHOD != 2 and not PIONRATIO and not (
        if origl != 1 and eff_mass_method != 2 and not (
                origl == 2 and eff_mass_method == 4):
            if matrix_subtraction:
                print("***ERROR***")
                print(
                    "dimension of GEVP matrix and start params do not match")
                print("(or for non-gevp fits, the start_param len>1)")
                sys.exit(1)
            else:
                assert (delta_e2_around_the_world is None and origl == 3)\
                    or (delta_e2_around_the_world is not None and origl == 4)

def prelimselect():
    """Select the preliminary fit function"""
    def prefit_func(ctime, trial_params):
        """initial function; blocked"""
        assert None
        if ctime or trial_params:
            pass
    return prefit_func

def const_plus_exp(start_params):
    """const (eff energy) + exp (systematic) fit"""
    def prefit_func(ctime, trial_params):
        """eff mass method 1, fit func, single const fit
        """
        return [trial_params[2*i]+trial_params[2*i+1]*np.exp(-(
            trial_params[-1]-trial_params[2*i])*ctime)\
                for i in range(int((len(start_params)-1)/2))]
    return prefit_func

def mod_start_params(start_params, sys_energy_guess):
    """Modify start parameters for effective mass systematic energy fit"""
    start_params.append(sys_energy_guess)
    assert not (
        len(start_params)-1) % 2, \
        "bad start parameter spec:"+str(start_params)
    return start_params

def three_asserts(gevp, matrix_subtraction, noatwsub):
    """three asserts"""
    assert gevp
    assert not matrix_subtraction
    assert noatwsub

def atwfit(delta_e2_around_the_world, delta_e_around_the_world,
           start_params, dim, lent):
    """estimate around the world via fit"""
    assert delta_e2_around_the_world is None
    assert len(range(int((len(start_params)-1)/3))) == dim
    def prefit_func(ctime, trial_params):
        """eff mass method 1, fit func, single const fit
        """
        rrl = range(dim)
        term1 = [trial_params[3*i] for i in rrl]
        term2 = [trial_params[3*i+1]*exp(
            -(trial_params[-1]-trial_params[3*i])*ctime)
                 for i in rrl]
        term3 = [trial_params[3*i+2]*exp(
            -1*lent*misc.massfunc())*exp(
                -1*ctime*delta_e_around_the_world)
                 for i in rrl]
        #print(term1, term2, term3)
        return [term1[i]+term2[i]+term3[i] for i in rrl]
    return prefit_func


def atwfit_second_order(start_params, dim, lent,
                        delta_e_around_the_world, mine2):
    """continue the selection process
    """
    assert len(range(int((len(start_params)-1)/4))) == dim
    def prefit_func(ctime, trial_params):
        """eff mass method 1, fit func, single const fit
        """
        rrl = range(dim)
        term1 = [trial_params[4*i] for i in rrl]
        term2 = [trial_params[4*i+1]*exp(
            -(trial_params[-1]-trial_params[4*i])*ctime)
                 for i in rrl]
        term3 = [trial_params[4*i+2]*exp(-1*lent*misc.massfunc())*exp(
            -1*ctime*(trial_params[4*i]-delta_e_around_the_world))
                 for i in rrl]
        term4 = [trial_params[4*i+3]*exp(-1*lent*mine2)*exp(-1*ctime*(
            trial_params[4*i]-delta_e_around_the_world)) for i in rrl]
        #print(term1, term2, term3, term4)
        return [term1[i]+term2[i]+term3[i]+term4[i] for i in rrl]
    return prefit_func


def constfit(len_start_params, rescale=1.0):
    """prefit function just returns the trial params"""
    if len_start_params == 1.0:
        if rescale != 1.0:
            def prefit_func(_, trial_params):
                """eff mass method 1, fit func, single const fit
                """
                return rescale*trial_params
        else:
            def prefit_func(_, trial_params):
                """eff mass method 1, fit func, single const fit
                """
                return trial_params
    else:
        if rescale != 1.0:
            def prefit_func(_, trial_params):
                """eff mass method 1, fit func, single const fit
                """
                return [rescale*trial_param for trial_param in trial_params]
        else:
            def prefit_func(_, trial_params):
                """eff mass method 1, fit func, single const fit
                """
                return trial_params
    return prefit_func

def eff_mass_3_func(lenparams, rescale, fits, tvecs):
    """fit function for effective mass method 3"""
    origl, mult = lenparams
    add_const_vec, lt_vec = tvecs

    if rescale != 1.0:
        def prefit_func(ctime, trial_params):
            """eff mass 3, fit func, rescaled
            """
            return [rescale * fits.fid['fit_func_1p'][
                add_const_vec[j]](ctime, trial_params[j:j+1*origl],
                                  lt_vec[j])
                    for j in range(mult)]
    else:
        def prefit_func(ctime, trial_params):
            """eff mass 3, fit func, rescaled
            """
            return [fits.fid['fit_func_1p'][add_const_vec[j]](
                ctime, trial_params[j:j+1*origl], lt_vec[j])
                    for j in range(mult)]
    return prefit_func

def fit_func_die():
    """exit; bad fit function selection"""
    print("***ERROR***")
    print("check config file fit func selection.")
    sys.exit(1)

def expfit_gevp(lenparams, fit, rescale, fits, tvecs):
    """get prefit function if not fitting effective mass
    and not using GEVP
    """
    origl, mult = lenparams
    add_const_vec, lt_vec = tvecs
    # check len of start params
    if origl != 2 and fit:
        print("***ERROR***")
        print("flag 1 length of start_params invalid")
        sys.exit(1)
    # select fit function
    if rescale != 1.0:
        def prefit_func(ctime, trial_params):
            """gevp fit func, non eff mass"""
            return [
                rescale*fits.fid['fit_func_exp_gevp'][add_const_vec[j]](
                    ctime, trial_params[j*origl:(j+1)*origl], lt_vec[j])
                for j in range(mult)]
    else:
        def prefit_func(ctime, trial_params):
            """gevp fit func, non eff mass"""
            return [fits.fid['fit_func_exp_gevp'][add_const_vec[j]](
                ctime, trial_params[j*origl:(j+1)*origl], lt_vec[j])
                    for j in range(mult)]
    return prefit_func

def expfit(fit, origl, add_const, rescale, fits):
    """select a fit function if not fitting effective mass
    and not using gevp
    """
    if fit:
        # check len of start params
        if origl != (3 if add_const else 2):
            print("***ERROR***")
            print("flag 2 length of start_params invalid")
            sys.exit(1)
        # select fit function
        if rescale != 1.0:
            def prefit_func(ctime, trial_params):
                """Rescaled exp fit function."""
                return rescale*fits.fid[
                    'fit_func_exp'](ctime, trial_params)
        else:
            def prefit_func(ctime, trial_params):
                """Prefit function, copy of
                exponential fit function."""
                return fits.use('fit_func_exp')(ctime, trial_params)
    else:
        def prefit_func(__, _):
            """fit function doesn't do anything because FIT = False"""
            pass
    return prefit_func

# DO NOT EDIT BELOW THIS LINE
# for general pencil of function

def pencil_mod(prefit_func, fit, num_pencils, rescale, start_params):
    """Modify the fit function for GPOF method"""
    if fit:
        fit_func_copy = copy(prefit_func)

    if num_pencils > 0:
        def fit_func(ctime, trial_params):
            """Fit function (num_pencils > 0)."""
            return np.hstack(
                [rescale*fit_func_copy(
                    ctime, trial_params[i*len(start_params):(i+1)*len(
                        start_params)]) for i in range(2**num_pencils)])
    else:
        def fit_func(ctime, trial_params):
            """Fit function."""
            return prefit_func(ctime, trial_params)
    return fit_func
