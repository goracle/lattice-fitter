"""test if file exists"""
import sys

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
