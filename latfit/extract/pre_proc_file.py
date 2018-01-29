"""test if file exists"""
import sys
import numpy as np

from latfit.config import STYPE

if STYPE == 'ascii':
    def pre_proc_file(ifile, input_f):
        """Try to open file, print error if not found"""
        ifile = input_f + "/" + ifile
        try:
            open(ifile, "r")
        except TypeError:
            print("Either domain is invalid,", "or folder is invalid.")
            print("Double check contents of folder.")
            print("Offending file(s):")
            print(ifile)
            sys.exit(1)
        return ifile

elif STYPE == 'hdf5':
    def pre_proc_file(ifile, input_f):
        """ifile is now an array. Do a meaningless check.
        (could be meaningful later)"""
        if not isinstance(ifile,
                          np.ndarray) or not isinstance(ifile[0], np.complex):
            print("***ERROR***")
            print("The samples are not in the right format")
            print("File/sample mismatch:")
            print("File:", input_f)
            print("sample:", ifile)
            sys.exit(1)
        return ifile
