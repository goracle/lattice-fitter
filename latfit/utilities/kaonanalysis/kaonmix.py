"""Does the kaon mix3/4 uv divergence subtraction"""

from collections import defaultdict
import numpy as np
import kaonpostproc as kpp
import kaonfileproc as kfp
import kaonprojop
from latfit.utilities.h5jack import LT as LT_CHECK
from latfit.utilities.h5jack import dojackknife

def mix_coeffs(type4, mix4, trajl, fidx):
    """Extract the jackknifed alpha_i
    == <|Q_i|K>/<|sbar G5 d|> = <type4_bubble>/<mix4>,
    jackknife the results"""
    ltraj = len(trajl)
    alpha = {}

    # for reference
    # shapeT4 = (ltraj, 8, 4, LT_CHECK, LT_CHECK)
    # shapeM4 = (ltraj, 2, LT_CHECK, LT_CHECK)

    alpha = defaultdict(lambda: np.zeros((ltraj, LT_CHECK), dtype=np.complex))

    for i in np.arange(1, 11): # assumes we've averaged over tk for type4 and mix4

        # project onto Qi
        projtemp = np.zeros((ltraj, LT_CHECK), dtype=np.complex)
        for num, _ in enumerate(trajl):
            projtemp[num] = kaonprojop.qi_proj_type4(
                type4[num], i)

        # jackknife the results
        projtemp = dojackknife(projtemp)
        jackmix = dojackknife(mix4[:, fidx, :])

        for num, _ in enumerate(trajl):
            alpha[str(i)][num] = projtemp[num]/jackmix[num]
    return alpha



def mix_subtract(alpha, mix3, mix4tox, otype, ltraj):
    """Subtract mix diagrams, assumes we've jackknifed the operators"""

    # mix3 = Isospin0ProjMix3(mix3) # useless
    # mix4tox = Isospin0ProjMix4(mix4tox) # useless

    # shape == key, ltraj, fidx, tdis

    for i in np.arange(1, 11):

        if otype == 'sigma':
            for momdiag in mix3:
                assert mix3[momdiag].shape == (ltraj, 2, LT_CHECK),\
                    "bad mix3 shape"
                kpp.QOP_SIGMA[str(i)][kfp.gen_key(momdiag)] -= alpha[
                    str(i)]*mix3[momdiag][:, 0, :] # 0 for pseudoscalar vertex
            for momdiag in mix4tox:
                assert mix4tox[momdiag].shape == (ltraj, 2, LT_CHECK),\
                    "bad mix4 shape"
                kpp.QOP_SIGMA[str(i)][kfp.gen_key(momdiag)] -= alpha[
                    str(i)]*mix4tox[momdiag][:, 0, :]

        elif otype == 'pipi':
            for momdiag in mix3:
                assert mix3[momdiag].shape == (ltraj, 2, LT_CHECK),\
                    "bad mix3 shape"
                kpp.QOPI0[str(i)][kfp.gen_key(momdiag)] -= alpha[
                    str(i)]*mix3[momdiag][:, 0, :] # 0 for pseudoscalar vertex
            for momdiag in mix4tox:
                assert mix4tox[
                    momdiag].shape == (ltraj, 2, LT_CHECK), "bad mix4 shape"
                kpp.QOPI0[str(i)][kfp.gen_key(momdiag)] -= alpha[
                    str(i)]*mix4tox[momdiag][:, 0, :]

        else:
            assert None, "k->pi not written yet"
