"""test if file exists"""
import sys

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
        if not ifile in input_f:
            print("***ERROR***")
            print("The samples did not come from this hdf5 file")
            print("File/sample mismatch:")
            print("File:", input_f)
            print("sample:", ifile)
            sys.exit(1)
        return ifile
