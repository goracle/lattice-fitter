#!/usr/bin/python3
"""Sum tsrc and bin"""
import re
import os.path
from os import listdir
from os.path import isfile, join
import numpy as np
import avgvac as avac
import jk_make as jk
import read_file as rf

def proc_file(filename, sum_tsrc=True):
    """gets the array from the file
    also gets the GEVP transpose array
    """
    len_t = rf.find_dim(filename)
    if not len_t:
        return None
    front = np.zeros(shape=(len_t, len_t), dtype=complex)
    #back = front
    filen = open(filename, 'r')
    for line in filen:
        lsp = line.split()
        if len(lsp) != 4:
            return None
        #lsp[0] = tsrc, lsp[1] = tdis
        tsrc = int(lsp[0])
        #tsnk = (int(lsp[0])+int(lsp[1]))%len_t
        tdis = int(lsp[1])
        #tdis2 = len_t-int(lsp[1])-1
        front.itemset(tsrc, tdis, complex(float(lsp[2]), float(lsp[3])))
        #back.itemset(tsnk, tdis2, complex(float(lsp[2]), float(lsp[3])))
    if sum_tsrc:
        #return sum_rows(front), sum_rows(back)
        retarr = rf.sum_rows(front, True)
    else:
        retarr = front
    return retarr
        #return front, back

def call_sum(filen, dur, binsize=1, bin_num=1, already_summed=False):
    """get the output file name, data
    """
    if binsize != 1:
        traj = rf.traj(filen)
        if re.search(r'traj_(\d+)B(\d+)_', traj):
            print("Skipping. File to process is already binned:", filen)
            outfile = None
        else:
            outfile = re.sub(str(traj), str(binsize)+'B'+str(bin_num), filen)
    else:
        outfile = filen
    #we've obtained the output file name, now get the data
    if not outfile:
        data = None
    elif os.path.isfile(dur+outfile):
        print("Skipping:", filen, "File exists.")
        data = None
    else:
        if not already_summed:
            data = proc_file(filen, True)
        else:
            data = avac.proc_vac(filen)
        if data is None:
            print("Skipping file", filen, "should be 4 numbers per line, non-4 value found.")
    return data, outfile

def inter_sum(already_summed, onlyfiles):
    """Sum tsrc"""
    if not already_summed:
        dur = 'summed_tsrc_diagrams/'
        if not os.path.isdir(dur):
            os.makedirs(dur)
    else:
        return
    for filen in onlyfiles:
        data, outfile = call_sum(filen, dur, 1)
        if data and outfile:
            rf.write_vec_str(data, dur+outfile)
    print("Done writing files averaged over tsrc.")

def get_baselist(onlyfiles):
    """Get list of base names from diagram file names."""
    baselist = {}
    for filen in onlyfiles:
        base = jk.base_name(filen)
        try:
            traj = int(rf.traj(filen))
        except TypeError:
            continue
        #throw out trajectories not thermalized
        if traj < 1000:
            continue
        if not base in baselist:
            baselist[base] = [(filen, rf.traj(filen))]
        else:
            baselist[base].append((filen, rf.traj(filen)))
    return baselist

def bin_tsrc_sum(binsize, step, already_summed=False):
    """Bin diagrams"""
    nmax = int(binsize)/int(step)
    onlyfiles = [f for f in listdir('.') if isfile(join('.', f))]
    if nmax == 1:
        inter_sum(already_summed, onlyfiles)
        return
    else:
        if not already_summed:
            dur = 'summed_tsrc_diagrams/binned_diagrams/binsize'+str(binsize)+'/'
        else:
            dur = 'binned_diagrams/binsize'+str(binsize)+'/'
        if not os.path.isdir(dur):
            os.makedirs(dur)
        baselist = get_baselist(onlyfiles)
        for base in baselist:
            count = 0
            bin_num = 0
            data = None
            print("Processing base", base)
            #sort by trajectory number
            blist = np.array(
                sorted(baselist[base], key=lambda tup: int(tup[1])))
            for filen in blist[:, 0]:
                odata, outfile = call_sum(
                    filen, dur, binsize, bin_num, already_summed)
                if odata is None or not outfile:
                    continue
                if data is None:
                    data = np.array(odata)
                else:
                    data += np.array(odata)
                print("Processed", filen)
                count += 1
                if count % nmax == 0:
                    print("Accumulated data.  Writing binned diagram.")
                    data /= nmax
                    rf.write_vec_str(data, dur+outfile)
                    bin_num += 1
                    count = 0
                    data = None
        print("Done writing files binned with bin size =", binsize)
        return

def main():
    """Sum tsrc and block"""
    #global params, set by hand
    mdstep = int(input("Please enter average non-blocked separation."))
    as1 = input("Already summed? y/n")
    if as1 == 'y':
        already_summed = True
    elif as1 == 'n':
        already_summed = False
    else:
        sys.exit(1)
    #end global params
    for binsize in [mdstep, 20, 40, 60, 80]:
        if binsize == mdstep:
            if not already_summed:
                print("Averaging diagrams over tsrc.")
            else:
                continue
        else:
            print("Doing binsize:", binsize)
        bin_tsrc_sum(binsize, mdstep, already_summed)

if __name__ == "__main__":
    main()
