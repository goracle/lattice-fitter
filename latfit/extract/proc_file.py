from __future__ import division

CSENT = object()
def proc_file(pifile, pjfile=CSENT):
    """Process the current file.
    Return covariance matrix entry I,indexj in the case of multi-file
    structure.
    Return the covariance matrix for single file.
    """
    #initialize return value named tuple. in other words:
    #create a type of object, rets, to hold return values
    #instantiate it with return values, then return that instantiation
    rets = namedtuple('rets', ['coord', 'covar'])
    if pjfile == CSENT:
        print "***ERROR***"
        print "Missing secondary file."
        sys.exit(1)
    #within true cond. of test, we assume number of columns is one
    with open(pifile) as ithfile:
        avgone = 0
        avgtwo = 0
        count = 0
        for line in ithfile:
            avgone += float(line)
            count += 1
        avgone /= count
        with open(pjfile) as jthfile:
            counttest = 0
            for line in jthfile:
                avgtwo += float(line)
                counttest += 1
            if not counttest == count:
                print "***ERROR***"
                print "Number of rows in paired files doesn't match"
                print count, counttest
                print "Offending files:", pifile, "and", pjfile
                sys.exit(1)
            else:
                avgtwo /= count
            #cov[I][indexj]=return value for folder-style
            #applying jackknife correction of (count-1)^2
            coventry = (count-1)/(1.0*count)*fsum([
                (float(l1)-avgone)*(float(l2)-avgtwo)
                for l1, l2 in izip(open(pifile), open(pjfile))])
            return rets(coord=avgone,
                        covar=coventry)
    print "***Unexpected Error***"
    print "If you\'re seeing this program has a bug that needs fixing"
    sys.exit(1)
    #delete me (if working)
            #append at position i,j to the covariance matrix entry
        #store precomputed covariance matrix (if it exists)
        #in convenient form
        #try:
        #COV = [[CDICT[PROCCOORDS[i][0]][PROCCOORDS[j][0]]
        #        for i in range(len(PROCCOORDS))]
        #       for j in range(len(PROCCOORDS))]
    #delete me end (if working)
