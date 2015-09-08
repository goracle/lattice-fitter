import os

from latfit.procargs import procargs

def inputexists(input):
    """Test to see if file/folder exists."""
    if not (os.path.isfile(input) or os.path.isdir(input)):
        print "File:", input, "not found"
        print "Folder:", input, "also not found."
        procargs(["h"])
    return 0
