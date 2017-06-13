def printerr(result_min, PARAM_ERR):
    for i in range(len(PARAM_ERR)):
        print("Minimized parameter #", i, " = ")
        print(result_min[i], "+/-", PARAM_ERR[i])
    return 0
        #ERR_A0 = sqrt(2*HINV[0][0])
        #ERR_ENERGY = sqrt(2*HINV[1][1])
        #print "a0 = ", result_min.x[0], "+/-", ERR_A0
        #print "energy = ", result_min.x[1], "+/-", ERR_ENERGY
