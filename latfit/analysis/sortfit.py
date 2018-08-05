"""Find the worst fit coordinates"""
import sys
import math
import numpy as np
from latfit.config import DISP_ENERGIES, ORIGL

def best_times(coord, cov, index, times):
    """minimize distance to dispersive (no interactions) line
    index is the operator index of the GEVP
    (todo:make more general?)
    chisq = num/denom
    """
    dist = []
    for i, ycoord in enumerate(coord):
        num = (ycoord-DISP_ENERGIES[index])**2
        denom = cov[i,i]
        dist.append([times[i], num/denom])
    dist = np.array(sorted(dist, key=lambda row: row[1]))
    return dist

def score_excl(excl1d, tsorted, lenfit, inversescore=True):
    """Sort the fit ranges based on which times are best
    (most likely to give good chi^2)
    score the excluded time slices for a particular (GEVP) dimension
    higher scores are better fits, so return -score
    so the better fits come first using sorted()
    """
    score = 0
    dof = lenfit-ORIGL-len(excl1d)
    for i, ttup in enumerate(tsorted):
        time, chisq = ttup
        if time in excl1d:
            score += chisq/dof
    if inversescore:
        score *= -1
    return score

def sortcombinations(combinations, tsorted, lenfit):
    return sorted(combinations, key=lambda row: score_excl(
        row, tsorted, lenfit))

# does not work
# the probablistic nature means we only have weak ordering
def sample_norms(sampler, tsorted, lenfit):
    """Score the powerset and find the corresponding
    normalized probabilities"""
    samp = sorted(list(sampler))
    total = 0
    probs = []
    for excl in samp:
        score = score_excl(excl, tsorted, lenfit, inversescore=False)
        probs.append(score)
        total += score
    norm = 1.0/total
    probs = np.array(probs)*norm
    assert np.allclose(
        np.sum(probs, axis=0), 1.0), "Probabilities chosen do not sum to 1."
    #for i, j in zip(samp, probs):
    #    print(i, j)
    #sys.exit(0)
    return probs, samp
