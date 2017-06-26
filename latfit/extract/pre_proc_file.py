import sys

def pre_proc_file(IFILE,INPUT):
    IFILE = INPUT + "/" + IFILE
    try:
        TRIAL = open(IFILE, "r")
    except TypeError:
        STR1 = "Either domain is invalid,"
        print(STR1, "or folder is invalid.")
        print("Double check contents of folder.")
        print("Offending file(s):")
        print(IFILE)
        sys.exit(1)
    return IFILE
