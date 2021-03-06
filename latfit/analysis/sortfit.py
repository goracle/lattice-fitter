"""Find the worst fit coordinates"""
import numpy as np
from latfit.config import DISP_ENERGIES, ORIGL
from latfit.utilities import exactmean as em

def best_times(coord, cov, index, times):
    """minimize distance to dispersive (no interactions) line
    index is the operator index of the GEVP
    (todo:make more general?)
    chisq (t^2) = num/denom
    """
    disp = np.asarray(DISP_ENERGIES)
    if hasattr(DISP_ENERGIES, '__iter__') and len(disp.shape) > 1:
        dispmean = em.acmean(DISP_ENERGIES, axis=1) if\
            np.asarray(DISP_ENERGIES).shape else 0
    elif hasattr(DISP_ENERGIES, '__iter__') and len(DISP_ENERGIES.shape) == 1:
        dispmean = DISP_ENERGIES
    else:
        dispmean = em.acmean(DISP_ENERGIES) if np.asarray(DISP_ENERGIES).shape else 0
    dist = []
    for i, ycoord in enumerate(coord):
        chisq = None
        if np.isnan(ycoord):
            chisq = 0
        else:
            try:
                num = (ycoord-DISP_ENERGIES[index])**2
            except IndexError:
                num = (ycoord-dispmean)**2
            denom = cov[i, i]
            assert not np.any(np.isnan(num)),\
                "difference (chisq (t^2) numerator)"+\
                " is not a number "+str(DISP_ENERGIES)+" "+str(ycoord)
            assert denom != 0, "Error: variance is 0"
            assert not np.any(np.isnan(num/denom)),\
                "chisq (t^2) is not a number."
            chisq = em.acmean(num/denom)
        dist.append([times[i], chisq])
    dist = np.array(sorted(dist, key=lambda row: row[1]))
    return dist

def score_excl(excl1d, tsorted, lenfit, inversescore=True):
    """Sort the fit ranges based on which times are best
    (most likely to give good chi^2 (t^2))
    score the excluded time slices for a particular (GEVP) dimension
    higher scores are better fits, so return -score
    so the better fits come first using sorted()
    """
    score = 0
    dof = lenfit-ORIGL-len(excl1d)
    assert dof > 0, "Degrees of freedom for sortfit < 1"
    for _, ttup in enumerate(tsorted):
        time, chisq = ttup
        if time in excl1d:
            score += chisq/dof
    if inversescore:
        score *= -1
    return score

def sortcombinations(combinations, tsorted, lenfit):
    """Sort the combinations of t using the sorted t list"""
    return sorted(combinations, key=lambda row: score_excl(
        row, tsorted, lenfit))

# does not work
# the probablistic nature means we only have weak ordering
def sample_norms(sampler, tsorted, lenfit):
    """Score the powerset and find the corresponding
    normalized probabilities"""
    samp = tuple(sorted(list(sampler)))
    total = 0
    probs = []
    assert samp, "no fit ranges"
    for excl in samp:
        score = score_excl(excl, tsorted, lenfit, inversescore=False)
        probs.append(score)
        total += score
    assert not np.isnan(total), "Score total is not a number."
    assert total != 0, "total is 0."
    norm = 1.0/total
    probs = np.array(probs)*norm
    assert np.allclose(
        em.acsum(probs, axis=0),
        1.0, rtol=1e-8),\
        "Probabilities chosen do not sum to 1:"+str(em.acsum(probs, axis=0))
    #for i, j in zip(samp, probs):
    #    print(i, j)
    return probs, samp
