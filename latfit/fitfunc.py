"""Get the actual fit function"""
# FIT FUNCTION/PROCESSING FUNCTION SELECTION

import latfit.analysis.misc as misc

# select which of the above library functions to use


def check_start_params_len(eff_mass, eff_mass_method, origl,
                           matrix_subtraction, delta_e2_around_the_world):
    """Perform a check on the length of the start parameters"""
    if eff_mass:

        # check len of start params
        #if ORIGL != 1 and EFF_MASS_METHOD != 2 and not PIONRATIO and not (
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

def prelimselect(eff_mass, eff_mass_method, rescale, start_params):
    """Select the fit function"""
    def prefit_func(ctime, trial_params):
        """initial function; blocked"""
        assert None
        if ctime or trial_params:
            pass
    if EFF_MASS:

        # select fit function
        if eff_mass_method == 1 or eff_mass_method == 2 or\
           eff_mass_method == 4:
            if rescale == 1.0:
                if len(start_params) == 1:
                    def prefit_func(_, trial_params):
                        """eff mass method 1, fit func, single const fit
                        """
                        return trial_params
    return prefit_func

#tbool = EFF_MASS and (EFF_MASS_METHOD == 1 or EFF_MASS_METHOD == 2 or EFF_MASS_METHOD == 4) and RESCALE == 1.0 and len(START_PARAMS) != 1
def prelim2(sys_energy_guess, prefit_func,
            start_params, delta_e2_around_the_world):
    """Continue the selection process"""
    bret = False
    if and sys_energy_guess is not None:
        bret = True
        start_params.append(sys_energy_guess)
        assert not (
            len(start_params)-1) % 2, \
            "bad start parameter spec:"+str(start_params)
        if not (len(start_params)-1) % 2 and\
            delta_e2_around_the_world is None:
            def prefit_func2(ctime, trial_params):
                """eff mass method 1, fit func, single const fit
                """
                return [trial_params[2*i]+trial_params[
                    2*i+1]*exp(-(
                        trial_params[-1]-trial_params[
                            2*i])*ctime) for i in range(
                                int((len(start_params)-1)/2))]
            prefit_func = prefit_func2
    return (start_params, prefit_func, bret)

def prelim3(gevp, matrix_subtraction, noatwsub):
    """three asserts"""
    assert GEVP
    assert not MATRIX_SUBTRACTION
    assert NOATWSUB

# if tbool:
def prelim4(delta_e2_around_the_world, delta_e_around_the_world, start_params, dim):
    """something"""
    # estimate around the world via fit
    assert DELTA_E2_AROUND_THE_WORLD is None
    assert len(range(int((len(START_PARAMS)-1)/3))) == DIM
    def prefit_func(ctime, trial_params):
        """eff mass method 1, fit func, single const fit
        """
        rrl = range(DIM)
        term1 = [trial_params[3*i] for i in rrl]
        term2 = [trial_params[3*i+1]*exp(
            -(trial_params[-1]-trial_params[3*i])*ctime)
                    for i in rrl]
        term3 = [trial_params[3*i+2]*exp(
            -1*LT*misc.massfunc())*exp(
                -1*ctime*DELTA_E_AROUND_THE_WORLD)
                    for i in rrl]
        #print(term1, term2, term3)
        return [term1[i]+term2[i]+term3[i] for i in rrl]
    return prefit_func

def prelim5(start_params, dim, delta_e_around_the_world, mine2):
    """continue the process
    """
    assert len(range(int((len(START_PARAMS)-1)/4))) == DIM
    def prefit_func(ctime, trial_params):
        """eff mass method 1, fit func, single const fit
        """
        rrl = range(DIM)
        term1 = [trial_params[4*i] for i in rrl]
        term2 = [trial_params[4*i+1]*exp(
            -(trial_params[-1]-trial_params[4*i])*ctime)
                    for i in rrl]
        term3 = [trial_params[4*i+2]*exp(-1*LT*misc.massfunc())*exp(
                -1*ctime*(trial_params[4*i]-DELTA_E_AROUND_THE_WORLD))
                    for i in rrl]
        term4 = [trial_params[4*i+3]*exp(-1*LT*MINE2)*exp(-1*ctime*(
            trial_params[4*i]-DELTA_E_AROUND_THE_WORLD)) for i in rrl]
        #print(term1, term2, term3, term4)
        return [term1[i]+term2[i]+term3[i]+term4[i] for i in rrl]
    return prefit_func
                else:
                    def prefit_func(_, trial_params):
                        """eff mass method 1, fit func, single const fit
                        """
                        return trial_params
        else:
            if len(START_PARAMS) == 1:
                def prefit_func(_, trial_params):
                    """eff mass method 1, fit func, single const fit
                    """
                    return RESCALE*trial_params
            else:
                def prefit_func(_, trial_params):
                    """eff mass method 1, fit func, single const fit
                    """
                    return [RESCALE*trial_param for
                            trial_param in trial_params]

    elif EFF_MASS_METHOD == 3:
        if RESCALE != 1.0:
            def prefit_func(ctime, trial_params):
                """eff mass 3, fit func, rescaled
                """
                return [RESCALE * FITS.f['fit_func_1p'][
                    ADD_CONST_VEC[j]](ctime, trial_params[j:j+1*ORIGL],
                                      LT_VEC[j])
                        for j in range(MULT)]
        else:
            def prefit_func(ctime, trial_params):
                """eff mass 3, fit func, rescaled
                """
                return [FITS.f['fit_func_1p'][ADD_CONST_VEC[j]](
                    ctime, trial_params[j:j+1*ORIGL], LT_VEC[j])
                        for j in range(MULT)]
    else:
        print("***ERROR***")
        print("check config file fit func selection.")
        sys.exit(1)

else:
    if GEVP:
        # check len of start params
        if ORIGL != 2 and FIT:
            print("***ERROR***")
            print("flag 1 length of start_params invalid")
            sys.exit(1)
        # select fit function
        if RESCALE != 1.0:
            def prefit_func(ctime, trial_params):
                """gevp fit func, non eff mass"""
                return [
                    RESCALE*FITS.f['fit_func_exp_gevp'][ADD_CONST_VEC[j]](
                        ctime, trial_params[j*ORIGL:(j+1)*ORIGL], LT_VEC[j])
                    for j in range(MULT)]
        else:
            def prefit_func(ctime, trial_params):
                """gevp fit func, non eff mass"""
                return [FITS.f['fit_func_exp_gevp'][ADD_CONST_VEC[j]](
                    ctime, trial_params[j*ORIGL:(j+1)*ORIGL], LT_VEC[j])
                        for j in range(MULT)]
    else:
def noeff_mass_nogevp(fit, origl, add_const, rescale, fits):
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
                return rescale*fits.f[
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

