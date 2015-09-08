from latfit.procargs import procargs

def numex_err(nextra, switch):
    """Check nextra (the number of extra parameters) for errors.
    Return NUMPEXTRA, the number of extra parameters.
    """
    NUMPEXTRA = -1
    if isinstance(nextra, str):
        try:
            OPSTEMP = nextra
            OPSTEMP = int(OPSTEMP)
        except ValueError:
            print "***ERROR***"
            print "Invalid number of extra parameters."
            print "Expecting an int >= 0."
            procargs(["h"])
        if OPSTEMP >= 0:
            NUMPEXTRA = OPSTEMP
        else:
            NUMPEXTRA = -1
    if switch == '0':
        if NUMPEXTRA == -1:
            print "Input the number of extra fit parameters desired / 2"
            print "Two extra parameters will be used for each Extra"
            print "E.g., if you input 3, the number of extra parameters will"
            print "be 6.  See definition of pade given in source."
            print "Please fix to be more general:"
            print "each b_i value is taken to be greater than 4*(m_pi)^2"
            print "Extra = "
            NUMPEXTRA = int(raw_input())
            if NUMPEXTRA < 0 or (not isinstance(NUMPEXTRA, int)):
                print "***ERROR***"
                print "Expecting an int >= 0"
                sys.exit(2)
    return NUMPEXTRA
