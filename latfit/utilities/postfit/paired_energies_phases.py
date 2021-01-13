#!/usr/bin/python3
"""
 loop over energy levels
 for each energy, get a phase shift
 takes a log of latfit
 as the only input file
"""
import sys
import re

def dimof(log):
    """Get the GEVP dimension"""
    ret = None
    fn1 = open(log, 'r')
    for line in fn1:
        if 'file name of saved' in line:
            spl1 = line.split(' ')
            for it1 in spl1:
                if 'GEVP' not in it1:
                    continue
                spl2 = it1.split('x')
                if int(spl2[0]) == int(spl2[1][0]):
                    if ret is None:
                        ret = int(spl2[0])
                    else:
                        assert ret == int(spl2[0]), (ret, line)
                    break
    return ret

def build_list_enph(line, ens):
    """Build a list that looks like (e.g.)
    ['532(11)', '49.3(7.1)']
    from a line that looks like
    532(11) MeV ; phase shift (degrees): 49.3(7.1)
    """
    spl = line.split(' ')
    #ens = spl[0].rstrip()
    ens = ens.rstrip() # energy string
    phs = spl[-1].rstrip() # phase shift string
    if phs == '0(0)':
        phs = 'None'
    ret = [ens, phs]
    return ret

def geten(line):
    """Parse 'Loaded' line for energy, as in
    Loaded: <...> '0.524(11)']
    return '0.524(11)'
    """
    spl = line.split("'")
    ret = None
    for it1 in spl:
        if len(it1) < 2: # hack to prevent strings like '4'
            continue
        flag = 0
        for dig in it1:
            if dig in ('.', ')', '('):
                continue
            flag = 0
            try:
                int(dig)
            except ValueError:
                break
            flag = 1
        if flag:
            ret = it1
            break
    return ret


def parse_for_res(log):
    """Get line number of a pattern"""
    ret = {}
    pat = 'dimension of gevp of interest'
    pat2 = 'MeV ; phase shift (degrees)'
    patd = {'loaded':0, 'interest':0, 'started':0}
    dimofin = None
    with open(log) as ffile:
        #patd = reset_patd(patd)
        for num, line in enumerate(ffile):
            if 'Loaded' in line and 'energy' in line:
                en1 = geten(line)
                assert en1 is not None, line
                patd = reset_patd(patd)
                dimofin = None
                patd['loaded'] = 1
            elif pat in line:
                patd['interest'] = 1
                dimofin = int(line.split(':')[-1])
            elif pat2 in line and patd['loaded'] == 1:
                # check the log is following the
                # expected output pattern
                assert patd['interest'] == 1, (line, patd)
                if patd['started'] == 1:
                    count += 1
                else:
                    patd['started'] = 1
                    count = 0
                if dimofin == count:
                    toapp = build_list_enph(line, en1)
                    ret[dimofin] = toapp
                    patd = reset_patd(patd)
                    dimofin = None
                    count = 0
    return ret

def conv_dict_to_orderedlist(enphd):
    """Convert the dict to an ordered list"""
    ret = []
    count = 0
    for key in sorted(enphd.keys()):
        assert key == count, ("incomplete set of results",
                              enphd)
        count += 1
        ret.append(enphd[key])
    return ret

def reset_patd(patd):
    """Reset the pattern dictionary"""
    for i in patd:
        patd[i] = 0
    return patd

def phen(log):
    """For a particular slurm log,
    get the energy/phase pair"""
    dim = dimof(log)
    ret = []
    if dim is not None:
        ret = parse_for_res(log)
        ret = conv_dict_to_orderedlist(ret)
        assert dim == len(ret), (ret, dim)
    return ret

def paired_energies_phases(debug=False):
    """main"""
    ret = []
    for log in sys.argv[1:]:
        if debug:
            print('extracting energy/phase pairs in', log)
        pelist = phen(log)
        if pelist and debug:
            print('result:\n'+str(pelist))
        ret.append(pelist)
    if len(sys.argv) == 2:
        ret = ret[0]
    return ret


if __name__ == '__main__':
    print(paired_energies_phases())
