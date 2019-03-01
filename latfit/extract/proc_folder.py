"""get file from folder"""
import re
import os
import sys
import numpy as np
import h5py
import linecache

from latfit.config import GEVP
from latfit.config import STYPE
from latfit.config import GROUP_LIST
from latfit.config import BINNUM
from latfit.config import HALF, SUPERJACK_CUTOFF
from latfit.config import SLOPPYONLY
from latfit.mathfun.elim_jkconfigs import elim_jkconfigs
import latfit
import latfit.analysis.misc as misc

def binout(out):
    """Reduce the number of used configs
    """
    lout = len(out)
    
    while len(out)%BINNUM != 0:
        out = elim_jkconfigs(out, [len(out)-1])
        lout = len(out)
        assert lout > SUPERJACK_CUTOFF,\
            "total amount of configs must be >= exact configs"
        #out = out[:-1]
    return out

if STYPE == 'hdf5':
    def open_dataset(prefix, hdf5_file, ctime, fn1):
        """ open dataset given a group prefix
        return a numpy array
        """
        return np.array(fn1[prefix+'/'+hdf5_file.split('.')[
            0]][:, ctime])

    def proc_folder(hdf5_file, ctime, other_regex="", opdim=(None, None)):
        """Get data from hdf5 file (even though it's called proc_folder)"""
        if other_regex:
            pass
        try:
            fn1 = h5py.File(hdf5_file, 'r')
        except OSError:
            fn1 = h5py.File(hdf5_file.split('/')[-1], 'r')
        try:
            out = np.array(fn1[hdf5_file][:, ctime])
        except KeyError:
            try:
                out = np.array(fn1[hdf5_file.split('.')[0]][:, ctime])
            except KeyError:
                if proc_folder.sent:
                    proc_folder.sent2 = 0
                    proc_folder.prefix = test_prefix(
                        hdf5_file, ctime, fn1, GROUP_LIST)
                    latfit.config.TITLE_PREFIX = proc_folder.prefix + \
                        ' ' + latfit.config.TITLE_PREFIX
                    proc_folder.sent = 0
                try:
                    out = open_dataset(proc_folder.prefix,
                                       hdf5_file, ctime, fn1)
                except KeyError:
                    print("***ERROR***")
                    print("Check the hdf5 prefix.  dataset cannot be found.")
                    print("dataset name:",
                          proc_folder.prefix+'/'+hdf5_file.split('.')[0])
                    sys.exit(1)
        out = elim_jkconfigs(out)
        out = halftotal(out, ctime=ctime, opdim=opdim)
        #out = halftotal(out, ctime=ctime,override='first half')
        #out = halftotal(out, ctime=ctime,override='second half')
        #out = halftotal(out, ctime=ctime,override='first half')
        #out = halftotal(out, ctime=ctime,override='first half')
        #out = halftotal(out, ctime=ctime,override='first half')
        out = binout(out)
        return out
    proc_folder.sent = object()
    proc_folder.prefix = GROUP_LIST[0]

    def roundtozero(arr, ctime, opdim=(None, None)):
        """If the correlator is close to 0,
        zero it
        """
        arr = np.asarray(arr)
        larr = len(arr)
        assert larr > 1
        err = np.std(arr, ddof=1, axis=0)*np.sqrt(len(arr)-1)
        avg = np.mean(arr, axis=0)
        assert not hasattr(avg, '__iter__')
        if abs(avg) < err and opdim[0] == opdim[1] and opdim[0] is not None:
            print("setting correlator on time slice",
                  ctime, "with operator dimensions", opdim, "to 0.")
            ret = np.zeros(larr, dtype=np.complex), True
            assert ret[0].shape == np.asarray(arr).shape
        else:
            ret = arr, False
        return ret
            

    def halftotal(out, override=None, ctime=None, opdim=(None, None)):
        """First half second half analysis
        """
        sloppy = out[SUPERJACK_CUTOFF:]
        sloppy = half(sloppy, override)
        # check the sloppy blocks since the error bar is more reliable
        # sloppy, didround = roundtozero(sloppy, ctime, opdim=opdim)
        didround = False
        assert isinstance(didround, bool)
        if SUPERJACK_CUTOFF:
            exact = out[:SUPERJACK_CUTOFF]
            exact = half(exact, override)
            exact = np.asarray(exact)
            # if we did round down, round down the exact blocks too
            if didround:
                exact = np.zeros(exact.shape, np.complex)
        if SLOPPYONLY or not SUPERJACK_CUTOFF:
            ret = np.asarray(sloppy)
        else:
            ret = np.asarray([*exact, *sloppy])
        return ret


    def intceil(num):
        """Numpy returns a float when it should return an int for ceiling"""
        return int(np.ceil(num))

    def half(arr, override=None):
        """Take half of the array"""
        larr = len(arr)
        halfswitch = HALF if override is None else override
        if halfswitch == 'full':
            ret = arr
        elif halfswitch == 'first half':
            excl = np.array(range(len(arr)))[intceil(larr/2):]
            # ret = arr[:intceil(larr/2)]
        elif halfswitch == 'second half':
            excl = np.array(range(len(arr)))[:intceil(larr/2)]
            # ret = arr[intceil(larr/2):]
        elif halfswitch =='drop first quarter':
            excl = np.array(range(len(arr)))[:intceil(larr/4)]
        elif halfswitch =='drop second quarter':
            excl = np.array(
                range(len(arr)))[intceil(larr/4):2*intceil(larr/4)]
        elif halfswitch =='drop third quarter':
            excl = np.array(
                range(len(arr)))[2*intceil(larr/4):3*intceil(larr/4)]
        elif halfswitch =='drop fourth quarter':
            excl = np.array(
                range(len(arr)))[3*intceil(larr/4):]
        else:
            print("bad spec for half switch:", halfswitch)
            sys.exit(1)
        if halfswitch != 'full':
            excl = list(excl)
            ret = elim_jkconfigs(arr, excl)
        return ret

    def test_prefix(hdf5_file, ctime, fn1, alts=None):
        """Test hdf5 group name,
        try alternative names given in GROUP_LIST
        if initial guess doesn't work
        """
        retprefix = GROUP_LIST[0]
        if alts is None:
            print("***ERROR***")
            print("Empty hdf5 group prefix list in config.")
            print("Need at least one group prefix to try")
            sys.exit(1)
        for prefix in alts:
            try:
                open_dataset(prefix, hdf5_file, ctime, fn1)
                retprefix = prefix
            except KeyError:
                continue
        return retprefix


elif STYPE == 'ascii':
    def proc_folder(folder, ctime, other_regex=""):
        """Process folder where blocks to be averaged are stored.
        Return file corresponding to current ensemble (lattice time slice).
        Assumes file is <anything>t<time><anything>
        Assumes only 1 valid file per match, e.g. ...t3... doesn't happen more
        than once.
        Both the int and float versions of ctime are treated the same.
        """
        # build regex as a string
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
            folder, [my_regex, my_regex2],
            [regex_reject1, regex_reject2], temp4, retname)
        # logic: if we found at least one match
        if retname != temp4:
            # logic: if we found >1 match
            if flag2 == 1:
                print("***ERROR***")
                print("File name collision.")
                print("Two (or more) files match the search.")
                print("Amend your file names.")
                print("Offending files:", retname)
                sys.exit(1)
        else:
            if not GEVP:
                print(debug[0])
                print(debug[1])
            print(folder)
            print("***ERROR***")
            print("Can't find file corresponding to x-value = ", ctime)
            print("regex = ", my_regex)
            sys.exit(1)
        if GEVP:
            ret = []
            with open(retname) as fn1:
                for i, l in enumerate(fn1):
                    line = linecache.getline(fn1, i+1)
                    ret.append(np.complex(float(line[0]), float(line[1])))
            ret = np.array(ret)
        else:
            ret = retname
        ret = halftotal(ret, opdim=opdim)
        ret = binout(ret)
        return ret

    def lookat_dir(folder, my_regex, regex_reject, temp4, retname):
        """loop on dir, look for file we want"""
        flag2 = 0
        debug = [None, None]
        for root, dirs, files in os.walk(folder):
            for name in files:
                # logic: if the search matches either int or float ctime
                # test for int match first
                if regex_reject[1] != "":
                    if re.search(my_regex[1], name) and (
                            not re.search(regex_reject[1], name)):
                        # logic: if we found another matching file in
                        # the folder already, then flag it
                        if retname != temp4:
                            flag2 = 1
                        retname = name
                # then test for a float match
                elif re.search(my_regex[0], name) and (
                        not re.search(regex_reject[0], name)):
                    # logic: if we found another matching file in
                    # the folder already, then flag it
                    if retname != temp4:
                        flag2 = 1
                    # logic: else save the file name to return
                    # after folder walk
                    retname = name
                else:
                    debug = [root, dirs]
        return retname, flag2, debug

misc.halftotal = halftotal
misc.binout = binout
