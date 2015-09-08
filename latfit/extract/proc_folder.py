import re
import os

def proc_folder(folder, ctime):
    """Process folder where blocks to be averaged are stored.
    Return file corresponding to current ensemble (lattice time slice).
    Assumes file is <anything>t<time><anything>
    Assumes only 1 valid file per match, e.g. ...t3... doesn't happen more
    than once.
    Both the int and float versions of ctime are treated the same.
    """
    #build regex as a string
    my_regex = r"t" + str(ctime)
    regex_reject1 = my_regex+r"[0-9]"
    regex_reject2 = ""
    flag2 = 0
    temp4 = object()
    retname = temp4
    if int(str(ctime-int(ctime))[2:]) == 0:
        my_regex2 = r"t" + str(int(ctime))
        regex_reject2 = my_regex2+r"[0-9]"
    temp1 = ""
    temp2 = ""
    for root, dirs, files in os.walk(folder):
        for name in files:
            #logic: if the search matches either int or float ctime
            #test for int match first
            if not regex_reject2 == "":
                if re.search(my_regex2, name) and (
                        not re.search(regex_reject2, name)):
                    #logic: if we found another matching file in
                    #the folder already, then flag it
                    if not retname == temp4:
                        flag2 = 1
                    retname = name
            #then test for a float match
            if re.search(my_regex, name) and (
                    not re.search(regex_reject1, name)):
                #logic: if we found another matching file in
                #the folder already, then flag it
                if not retname == temp4:
                    flag2 = 1
                #logic: else save the file name to return after folder walk
                retname = name
            else:
                #gather debugging information.
                temp1 = root
                temp2 = dirs
    #logic: if we found at least one match
    if not retname == temp4:
        if flag2 == 1:
            print "***ERROR***"
            print "File name collision."
            print "Two files match the search."
            print "Amend your file names."
            print "Offending files:", retname
            sys.exit(1)
        return retname
    print temp1
    print temp2
    print folder
    print "***ERROR***"
    print "Can't find file corresponding to x-value = ", ctime
    sys.exit(1)
