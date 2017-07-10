"""Get covariance matrix element, blocks, coordinates"""
from collections import namedtuple
import sys
import numpy as np

from latfit.mathfun.proc_meff import proc_meff
from latfit.mathfun.elim_jkconfigs import elim_jkconfigs
from latfit.extract.proc_line import proc_line

from latfit.config import UNCORR
from latfit.config import EFF_MASS
from latfit.config import START_PARAMS
from latfit.config import elim_jkconf_list

def proc_ijfile(ifile_tup, jfile_tup, reuse=None):
    """Process the current file.
    Return covariance matrix entry I, indexj in the case of multi-file
    structure.
    Return the covariance matrix for single file.
    """
    if reuse is None:
        reuse = {}
    rets = namedtuple('rets', ['coord', 'covar', 'returnblk'])

    #do i
    try:
        num_configs = len(reuse['i'])
    except TypeError:
        reuse['i'] = np.array([])
        getblock(ifile_tup, reuse, 'i')
        num_configs = len(reuse['i'])
    avgone = np.mean(reuse['i'], axis=0)

    #print num of configs
    if proc_ijfile.CONFIGSENT != 0:
        print("Number of configurations to average over:", num_configs)
        proc_ijfile.CONFIGSENT = 0

    #check if we're on the same block (i==j)
    sameblk = all(i == j for i, j in zip(ifile_tup, jfile_tup))

    if sameblk:
        reuse['j'] = reuse['i']
    else:
        #do j
        try:
            num_configs_test = len(reuse['j'])
        except TypeError:
            reuse['j'] = np.array([])
            getblock(jfile_tup, reuse, 'j')
            num_configs_test = len(reuse['j'])

        #check to make sure i, j have the same number of lines
        if not num_configs_test == num_configs:
            print("***ERROR***")
            print("Number of rows in paired files doesn't match")
            print(num_configs, num_configs_test)
            print("Offending files:", ifile_tup, "\nand", jfile_tup)
            sys.exit(1)
        if UNCORR:
            coventry = 0
    if not UNCORR:
        coventry = np.dot(reuse['i']-avgone, reuse['j']-np.mean(reuse['j']))
    return rets(coord=avgone, covar=coventry, returnblk=reuse['j'])

#test to see if we've already printed the number of configurations
proc_ijfile.CONFIGSENT = object()

if EFF_MASS:
    def getblock(file_tup, reuse, ij_str):
        """Given file tuple (for eff_mass),
        get block, store in reuse[ij_str]
        """
        for line, line2, line3 in zip(
                open(file_tup[0], 'r'),
                open(file_tup[1], 'r'),
                open(file_tup[2], 'r')):
            if not line+line2+line3 in reuse:
                reuse[line+line2+line3] = proc_meff(
                    line, line2, line3, file_tup)
            if reuse[line+line2+line3] == 0:
                reuse[line+line2+line3] = START_PARAMS[1]
            reuse[ij_str] = np.append(reuse[ij_str], reuse[line+line2+line3])
        if elim_jkconf_list:
            reuse[ij_str] = elim_jkconfigs(reuse[ij_str])

else:
    def getblock(ijfile, reuse, ij_str):
        """Given file, get block, store in reuse[ij_str]
        """
        for line in open(ijfile):
            reuse[ij_str] = np.append(reuse[ij_str], proc_line(line, ijfile))
        if elim_jkconf_list:
            reuse[ij_str] = elim_jkconfigs(reuse[ij_str])
