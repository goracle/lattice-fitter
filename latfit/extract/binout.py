"""Various functions to eliminate/bin configs"""
import sys
import numpy as np

from latfit.analysis.errorcodes import BoolThrowErr
import latfit.analysis.misc as misc

BINNUM = None
SUPERJACK_CUTOFF = None
SLOPPYONLY = BoolThrowErr()
HALF = None

def intceil(num):
    """Numpy returns a float when it should return an int for ceiling"""
    return int(np.ceil(num))

def elim_jkconfigs(ret, elimlist):
    """dummy version"""
    assert None
    if ret or elimlist:
        pass



def half(arr, override=None):
    """Take half of the array"""
    assert HALF is not None
    larr = len(arr)
    halfswitch = HALF if override is None else override
    ret = arr if halfswitch == 'full' else None
    if halfswitch == 'first half':
        excl = np.array(range(len(arr)))[intceil(larr/2):]
        # ret = arr[:intceil(larr/2)]
    elif halfswitch == 'second half':
        excl = np.array(range(len(arr)))[:intceil(larr/2)]
        # ret = arr[intceil(larr/2):]
    elif halfswitch == 'drop first quarter':
        excl = np.array(range(len(arr)))[:intceil(larr/4)]
    elif halfswitch == 'drop second quarter':
        excl = np.array(
            range(len(arr)))[intceil(larr/4):2*intceil(larr/4)]
    elif halfswitch == 'drop third quarter':
        excl = np.array(
            range(len(arr)))[2*intceil(larr/4):3*intceil(larr/4)]
    elif halfswitch == 'drop fourth quarter':
        excl = np.array(
            range(len(arr)))[3*intceil(larr/4):]
    elif halfswitch == 'drop first eighth':
        excl = np.array(
            range(len(arr)))[:intceil(larr/8)]
    elif halfswitch == 'drop fourth eighth':
        excl = np.array(
            range(len(arr)))[3*intceil(larr/8):4*intceil(larr/8)]
    elif halfswitch == 'drop third eighth':
        excl = np.array(
            range(len(arr)))[2*intceil(larr/8):3*intceil(larr/8)]
    elif halfswitch == 'drop second eighth':
        excl = np.array(
            range(len(arr)))[intceil(larr/8):2*intceil(larr/8)]
    elif halfswitch != 'full':
        print("bad spec for half switch:", halfswitch)
        sys.exit(1)
    if halfswitch != 'full':
        excl = list(excl)
        ret = elim_jkconfigs(arr, excl)
    return ret

def binout(out):
    """Reduce the number of used configs
    """
    lout = len(out)

    while len(out)%BINNUM != 0:
        out = elim_jkconfigs(out, [len(out)-1])
        lout = len(out)
        assert lout > SUPERJACK_CUTOFF,\
            "total amount of configs must be >= exact configs"
        #out = out[:-1]
    return out

def halftotal(out, override=None):
    """First half second half analysis
    """
    sloppy = out[SUPERJACK_CUTOFF:]
    sloppy = half(sloppy, override)
    # check the sloppy blocks since the error bar is more reliable
    # sloppy, didround = roundtozero(sloppy, ctime, opdim=opdim)
    didround = False
    assert isinstance(didround, bool)
    if SUPERJACK_CUTOFF:
        exact = out[:SUPERJACK_CUTOFF]
        exact = half(exact, override)
        exact = np.asarray(exact)
        # if we did round down, round down the exact blocks too
        if didround:
            exact = np.zeros(exact.shape, np.complex)
    if SLOPPYONLY or not SUPERJACK_CUTOFF:
        ret = np.asarray(sloppy)
    else:
        ret = np.asarray([*exact, *sloppy])
    return ret

misc.HALFTOTAL = halftotal
misc.BINOUT = binout
