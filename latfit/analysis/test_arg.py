"""Test argument to log"""
from warnings import warn

def test_arg(arg, sent=None):
    """Test if arg to log is less than zero (imaginary mass)
    """
    if arg <= 0 and sent != 0:
        #print("***ERROR***")
        warn("argument to log in eff. mass"+" calc is than 0: "+str(
            arg))
        print("argument to log in effective mass",
              "calc is less than 0:", arg)
        return False
    return True
