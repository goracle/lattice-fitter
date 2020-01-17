"""Make fit selections for final effective mass fits"""
import numpy as np

def invinc(inc, win):
    """Get excluded points from included and fit window"""
    ret = []
    swin = set(list(np.arange(win[0], win[1]+1)))
    for i in inc:
        toapp = swin-set(i)
        toapp = sorted(list(toapp))
        ret.append(toapp)
    return ret

# in order to get final effective mass plots

# p0, 32c, I=2
INCLUDE = [[6.0, 7.0, 8.0], [7.0, 9.0, 11.0], [8.0, 9.0, 10.0], [6.0, 7.0, 8.0, 9.0, 10.0]]
DIMSELECT = 3 # t-t0=1 dt1, phase
PARAM_OF_INTEREST = 'phase shift'
FIT_EXCL = invinc(INCLUDE, (6, 13))

INCLUDE = [[6.0, 7.0, 8.0], [6.0, 7.0, 8.0], [6.0, 8.0, 10.0], [6.0, 7.0, 8.0, 9.0, 10.0]]
DIMSELECT = 3 # t-t0=1 dt1, energy
PARAM_OF_INTEREST = 'energy'
FIT_EXCL = invinc(INCLUDE, (6, 10))

INCLUDE = [[6.0, 7.0, 8.0], [10.0, 11.0, 12.0], [6.0, 7.0, 8.0, 9.0, 10.0], [8.0, 9.0, 10.0]]
DIMSELECT = 2 # t-t0=1 dt1, both
PARAM_OF_INTEREST = None
FIT_EXCL = invinc(INCLUDE, (6, 13))

INCLUDE = [[6.0, 7.0, 8.0], [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
           [6.0, 8.0, 10.0], [8.0, 9.0, 10.0]]
DIMSELECT = 1 # t-t0=1 dt1, both
PARAM_OF_INTEREST = None
FIT_EXCL = invinc(INCLUDE, (6, 11))

INCLUDE = [[15.0, 16.0, 17.0], [16.0, 17.0, 18.0], [17.0, 18.0, 19.0], [14.0, 15.0, 16.0]]
DIMSELECT = 0 # t-t0=5 dt3, both
PARAM_OF_INTEREST = None
FIT_EXCL = invinc(INCLUDE, (14, 19))

# p1, 32c, I=2
INCLUDE = [[9.0, 10.0, 11.0, 12.0], [10.0, 11.0, 12.0], [9.0, 10.0, 11.0]]
DIMSELECT = 0 # t-t0=5, both
PARAM_OF_INTEREST = None
FIT_EXCL = invinc(INCLUDE, (9, 12))

# p1, 32c, I=2
INCLUDE = [[5.0, 9.0, 13.0], [5.0, 6.0, 7.0, 8.0, 9.0], [4.0, 5.0, 6.0]]
DIMSELECT = 1 # t-t0=1, both
PARAM_OF_INTEREST = None
FIT_EXCL = invinc(INCLUDE, (4, 13))

# p1, 32c, I=2
INCLUDE = [[5.0, 7.0, 9.0], [5.0, 6.0, 7.0], [4.0, 5.0, 6.0]]
DIMSELECT = 2 # t-t0=1, both
PARAM_OF_INTEREST = None
FIT_EXCL = invinc(INCLUDE, (4, 11))

DIMSELECT = None
PARAM_OF_INTEREST = None

# p11, 32c, I=2
INCLUDE = [[10.0, 11.0, 12.0], [11.0, 12.0, 13.0, 14.0, 15.0],
           [11.0, 13.0, 15.0], [9.0, 11.0, 13.0]]
DIMSELECT = 3 # t-t0=5, both
PARAM_OF_INTEREST = None
FIT_EXCL = invinc(INCLUDE, (9, 15))

# p11, 32c, I=2
INCLUDE = [[13.0, 14.0, 15.0], [11.0, 13.0, 15.0],
           [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], [9.0, 10.0, 11.0]]
DIMSELECT = 2 # t-t0=5, energy
PARAM_OF_INTEREST = 'energy'
FIT_EXCL = invinc(INCLUDE, (9, 15))

# p11, 32c, I=2
INCLUDE = [[12.0, 13.0, 14.0, 15.0], [13.0, 14.0, 15.0],
           [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], [10.0, 11.0, 12.0]]
DIMSELECT = 2 # t-t0=5, both
PARAM_OF_INTEREST = 'phase shift'
FIT_EXCL = invinc(INCLUDE, (9, 15))

# p11, 32c, I=2
INCLUDE = [[9.0, 10.0, 11.0], [8.0, 9.0, 10.0], [8.0, 9.0, 10.0],
           [9.0, 10.0, 11.0]]
DIMSELECT = 1 # t-t0=3, both
PARAM_OF_INTEREST = None
FIT_EXCL = invinc(INCLUDE, (8, 11))

# p11, 32c, I=2
INCLUDE = [[10.0, 12.0, 14.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0],
           [9.0, 10.0, 11.0]]
DIMSELECT = 0 # t-t0=5, both
PARAM_OF_INTEREST = None
FIT_EXCL = invinc(INCLUDE, (9, 15))

# p111, 32c, I=2
INCLUDE = [[8.0, 9.0, 10.0, 11.0], [9.0, 10.0, 11.0]]
DIMSELECT = 0 # t-t0=3, both
PARAM_OF_INTEREST = None
FIT_EXCL = invinc(INCLUDE, (8, 12))

# p111, 32c, I=2
INCLUDE = [[8.0, 9.0, 10.0, 11.0], [6.0, 10.0, 14.0]]
DIMSELECT = 1 # t-t0=1, both
PARAM_OF_INTEREST = None
FIT_EXCL = invinc(INCLUDE, (6, 17))

# p0, 24c, I=2
INCLUDE = [[9.0, 11.0, 13.0], [9.0, 10.0, 11.0, 12.0], [10.0, 11.0, 12.0], [10.0, 11.0, 12.0]]
DIMSELECT = 0 # t-t0=4 dt2
PARAM_OF_INTEREST = None
FIT_EXCL = invinc(INCLUDE, (9, 13))

# default
INCLUDE = []
DIMSELECT = None
PARAM_OF_INTEREST = None

INCLUDE = tuple(INCLUDE)
FIT_EXCL = tuple(FIT_EXCL)

def print_include_messages(gevp):
    """Print messages"""
    if DIMSELECT is not None and gevp:
        print("dimension of gevp of interest:", DIMSELECT)
        if PARAM_OF_INTEREST is not None:
            print("param of gevp of interest:", PARAM_OF_INTEREST)
        else:
            print("param of gevp of interest: all")
