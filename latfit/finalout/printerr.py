"""Print error"""
import gvar
from latfit.config import VERBOSE:
def printerr(result_min, param_err):
    """Print the param error"""
    for i, err in enumerate(param_err):
        if VERBOSE:
            print("Minimized parameter #", i, " = ")
            print(gvar.gvar(result_min[i], err))

# ERR_A0 = sqrt(2*HINV[0][0])
# ERR_ENERGY = sqrt(2*HINV[1][1])
# print "a0 = ", result_min.x[0], "+/-", ERR_A0
# print "energy = ", result_min.x[1], "+/-", ERR_ENERGY

def avg_relerr(result_min, param_err):
    """Calculate the average relative error on the parameters"""
    relerr = 0
    for i, err in enumerate(param_err):
        relerr += err/result_min.x[i]
    relerr /= len(result_min.x)
    return relerr
