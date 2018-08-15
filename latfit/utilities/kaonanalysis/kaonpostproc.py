"""Storage for post processed operators.  After these are filled we write to disk"""
import h5py

QOPI0 = {} # operator dictionary
QOPI2 = {} # operator dictionary, I2
QOP_sigma = {} # sigma operator dictionary

for i in np.arange(1, 11):
    QOPI0[str(i)] = {} # momentum dictionary
    QOPI2[str(i)] = {} # momentum dictionary, I2
    QOP_sigma[str(i)] = {} # momentum dictionary

# structure is
# [<>]


def writeOut():
    """Write the result to file"""
    tseparr = []
    for i in np.arange(1, 11):
        for keyirr in QOPI0[str(i)]:
            tseparr.append(int(keyirr))
        assert tsep in QOPI0[str(i)], "tsep = "+keyirr+\
            " not in Q_"+str(i)+", I=0"
        assert tsep in QOPI2[str(i)], "tsep = "+keyirr+\
            " not in Q_"+str(i)+", I=2"
        assert tsep in QOP_sigma[str(i)], "tsep = "+keyirr+\
            " not in Q_"+str(i)+", sigma"
    for tsep in tseparr:
        QOPI0[str(i)]
