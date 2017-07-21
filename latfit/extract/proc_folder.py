"""get file from folder"""
import re
import os
import sys

from latfit.config import GEVP

def proc_folder(folder, ctime, other_regex=""):
    """Process folder where blocks to be averaged are stored.
    Return file corresponding to current ensemble (lattice time slice).
    Assumes file is <anything>t<time><anything>
    Assumes only 1 valid file per match, e.g. ...t3... doesn't happen more
    than once.
    Both the int and float versions of ctime are treated the same.
    """
    #build regex as a string
    if other_regex == "":
        my_regex = r"t" + str(ctime)
    else:
        my_regex = r"{0}{1}".format(other_regex, str(ctime))
    regex_reject1 = my_regex+r"[0-9]"
    regex_reject2 = ""
    flag2 = 0
    temp4 = object()
    retname = temp4
    if not isinstance(ctime, int):
        if int(str(ctime-int(ctime))[2:]) == 0:
            my_regex2 = r"t" + str(int(ctime))
            regex_reject2 = my_regex2+r"[0-9]"
    retname, flag2, debug = lookat_dir(
        folder, [my_regex, my_regex2], [regex_reject1, regex_reject2], temp4, retname)
    #logic: if we found at least one match
    if retname != temp4:
        #logic: if we found >1 match
        if flag2 == 1:
            print("***ERROR***")
            print("File name collision.")
            print("Two (or more) files match the search.")
            print("Amend your file names.")
            print("Offending files:", retname)
            sys.exit(1)
        return retname
    if not GEVP:
        print(debug[0])
        print(debug[1])
    print(folder)
    print("***ERROR***")
    print("Can't find file corresponding to x-value = ", ctime)
    print("regex = ", my_regex)
    sys.exit(1)

def lookat_dir(folder, my_regex, regex_reject, temp4, retname):
    """loop on dir, look for file we want"""
    flag2 = 0
    debug =[None, None]
    for root, dirs, files in os.walk(folder):
        for name in files:
            #logic: if the search matches either int or float ctime
            #test for int match first
            if regex_reject[1] != "":
                if re.search(my_regex[1], name) and (
                        not re.search(regex_reject[1], name)):
                    #logic: if we found another matching file in
                    #the folder already, then flag it
                    if retname != temp4:
                        flag2 = 1
                    retname = name
            #then test for a float match
            elif re.search(my_regex[0], name) and (
                    not re.search(regex_reject[0], name)):
                #logic: if we found another matching file in
                #the folder already, then flag it
                if retname != temp4:
                    flag2 = 1
                #logic: else save the file name to return after folder walk
                retname = name
            else:
                debug = [root, dirs]
    return retname, flag2, debug
