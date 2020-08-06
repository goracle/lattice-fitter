#!/usr/bin/python3
"""Generate proc_params.p files to set up latfit"""

import sys
import os
import pickle
import numpy as np

def main():
    """main"""
    print("proc_params.p key: MATRIX_SUBTRACTION, PR_GROUND_ONLY,",
          "PIONRATIO, STRONGCUTS, IRREP, LATTICE ENSEMBLE, DIM")
    print("enter 0 for False and 1 for True")
    if len(sys.argv) > 1:
        assert len(sys.argv) == 8, ("wrong number of proc params given,",
                                    "should be 7, received:", len(sys.argv))
        ret = {}
        ret['matsub'] = bool(int(sys.argv[1]))
        ret['pr ground only'] = bool(int(sys.argv[2]))
        ret['pion ratio'] = bool(int(sys.argv[3]))
        ret['strong cuts'] = bool(int(sys.argv[4]))
        ret['irrep'] = sys.argv[5]
        ret['lattice ensemble'] = sys.argv[6]
        ret['dim'] = int(sys.argv[7])
    if os.path.isfile('proc_params.p'):
        fn1 = open('proc_params.p', 'rb')
        ret = pickle.load(fn1)
        print("not writing new proc_params.p file,")
        print("proc_params.p exists with values:", ret)
    elif len(sys.argv) > 1:
        fn1 = open('proc_params.p', 'wb')
        print("writing proc_params.p with values:", ret)
        pickle.dump(ret, fn1)
        print("done.")
        

    

if __name__ == '__main__':
    main()
