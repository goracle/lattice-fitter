import numpy as np
from numpy import swapaxes

def maptomat(COV,DIMOPS=1):
    if DIMOPS==1:
        return COV
    else:
        Lt=len(COV)
        RETCOV=np.zeros((DIMOPS*Lt,DIMOPS*Lt))
        for i in range(Lt):
            for j in range(Lt):
                for a in range(DIMOPS):
                    for b in range(DIMOPS):
                        try:
                            RETCOV[i*DIMOPS+a][j*DIMOPS+b]=swapaxes(COV,1,2)[i][a][j][b]
                        except:
                            print("***ERROR***")
                            print("Dimension mismatch in mapping covariance tensor to matrix.")
                            print("Make sure time indices (i,j) and operator indices (a,b) are like COV[i][a][j][b].")
                            sys.exit(1)
        return RETCOV
