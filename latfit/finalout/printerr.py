"""Print error"""
def printerr(result_min, param_err):
    """Print the param error"""
    for i, err in enumerate(param_err):
        print("Minimized parameter #", i, " = ")
        print(result_min[i], "+/-", err)
    return 0
        #ERR_A0 = sqrt(2*HINV[0][0])
        #ERR_ENERGY = sqrt(2*HINV[1][1])
        #print "a0 = ", result_min.x[0], "+/-", ERR_A0
        #print "energy = ", result_min.x[1], "+/-", ERR_ENERGY
