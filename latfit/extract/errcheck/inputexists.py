"""Test to see if file/folder exists"""
import os

from latfit.procargs import procargs


def inputexists(input_f):
    """Test to see if file/folder exists.
    Print message help message otherwise
    """
    if not (os.path.isfile(input_f) or os.path.isdir(input_f)):
        print("File:", input_f, "not found")
        print("Folder:", input_f, "also not found.")
        procargs(["h"])
    return 0
