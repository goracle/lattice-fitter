"""Storage for post processed operators.  After these are filled we write to disk"""
QOPI0 = {} # operator dictionary
for i in range(10):
    QOPI0[str(i)] = {} # momentum dictionary
QOPI2 = {} # operator dictionary, I2
for i in range(10):
    QOPI2[str(i)] = {} # momentum dictionary, I2
QOP_sigma = {} # sigma operator dictionary
for i in range(10):
    QOP_sigma[str(i)] = {} # momentum dictionary

# structure is
# [<>]


def writeOut():
    """Write the result to file"""
    

