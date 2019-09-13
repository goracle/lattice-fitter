"""Container for operator projection (isospin coefficients)"""

from collections import defaultdict
import math
import numpy as np
from latfit.utilities.postprod.h5jack import LT as LT_CHECK
print("Imported projectkop with LT=", LT_CHECK)

def qi_proj_type1(pieces, i, isostr):
    """Project onto Q_i (k->pipi), type1"""
    ret = np.zeros((LT_CHECK), dtype=np.complex)

    #this is the reverse of every other class of diagrams
    # in this convention; part 1 is defined to be the sink
    gfactor = list(range(4))
    gfactor[0] = -1 # AV
    gfactor[1] = 1 if i in [5, 6, 7, 8] else -1 #V-A->V+A, VA
    gfactor[2] = 1 # VV
    gfactor[3] = -gfactor[1] # AA

    idxplus = (1 + i) % 2 # switch between color diagonal and mixed
    for tdis in range(LT_CHECK): # tdis distance to op
        for gcombidx, gfac in enumerate(gfactor):
            f_tdis = pieces[(0+idxplus, gcombidx, tdis)] # f
            fp_tdis = pieces[(2+idxplus, gcombidx, tdis)] # f'
            if i in [1, 2]:

                if isostr == 'I0': #  [ f+2f' ] *(-1)/sqrt(3)
                    ret[tdis] += (f_tdis+2*fp_tdis) * (-1)/(math.sqrt(3))

                elif isostr == 'I2': # [ -f+f' ] * 2/sqrt(6)
                    ret[tdis] += (-f_tdis+fp_tdis) * 2.0/(math.sqrt(6))

            elif i in [3, 4, 5, 6]:

                if isostr == 'I0': # -4/sqrt(3) f' - 2/sqrt(3) f = [
                    # f+2f' ] * (-2) /sqrt(3)
                    ret[tdis] += (f_tdis+2*fp_tdis) * 2.0/(-math.sqrt(3))

                elif isostr == 'I2': # 2*Q1 = [-f + f'] * 4/sqrt(6)
                    ret[tdis] += (-f_tdis+fp_tdis) * 4.0/(math.sqrt(6))

            elif i in [7, 8, 9, 10]: # ew penguin

                if isostr == 'I0': # 3/2 Q1 (e_d+e_u) =  [
                    # f+2f'] *(-1)/(2*sqrt(3))
                    ret[tdis] += (f_tdis+2*fp_tdis) * -1.0/(2*math.sqrt(3))

                elif isostr == 'I2': # 3/2 Q1 (e_d+e_u) = [
                    # -f + f'] * 1.0 / sqrt(6)
                    ret[tdis] += (-f_tdis+fp_tdis) * 1.0/(math.sqrt(6))

            ret[tdis] *= gfac # spin sign

    return ret
def get_gfac(i):
    """Get spin sign factor"""
    gfactor = list(range(4))
    gfactor[0] = 1 if i in [5, 6, 7, 8] else -1 #V-A->V+A, VA
    gfactor[1] = -1 # AV
    gfactor[2] = 1 # VV
    gfactor[3] = -gfactor[0] # AA
    return gfactor


def qi_proj_type2(pieces, i, isostr):
    """Project onto Q_i (k->pipi), type2"""
    ret = np.zeros((LT_CHECK), dtype=np.complex)
    gfactor = get_gfac(i)
    idxplus = (1 + i) % 2 # switch between color diagonal and mixed
    for tdis in range(LT_CHECK): # tdis distance to op
        for gidx, gfac in enumerate(gfactor):
            h_tdis = pieces[(0+idxplus, gidx, tdis)]
            hp_tdis = pieces[(2+idxplus, gidx, tdis)]
            if i in [1, 2]:

                if isostr == 'I0': # 3/sqrt(3) * h
                    ret[tdis] += h_tdis* 3/(math.sqrt(3))

                elif isostr == 'I2': # 0*h
                    ret[tdis] += 0

            elif i in [3, 4, 5, 6]:

                if isostr == 'I0': # 6/sqrt(3) * h - 3/sqrt(3) h' = [
                    # 2h-h' ] * 3/sqrt(3)
                    ret[tdis] += (2*h_tdis-hp_tdis) * 3.0/(math.sqrt(3))

                elif isostr == 'I2': # 2*Q1 = 2*0*h
                    ret[tdis] += 0


            elif i in [7, 8, 9, 10]: # ew penguin

                if isostr == 'I0': # 3/2 [
                    # Q1 (e_d+e_u) - h' * e_d * 3/sqrt(3) ] = [
                    # h + h' ] 3/(2*sqrt(3))
                    ret[tdis] += (h_tdis+hp_tdis)*3.0/2.0/math.sqrt(3)

                elif isostr == 'I2': # 3/2 Q1 (e_d+e_u) = 0 h
                    ret[tdis] += 0

            ret[tdis] *= gfac
    return ret


def qi_proj_sigma_type2(pieces, i):
    """Project the type2 sigma onto operator Qi"""
    ret = np.zeros((LT_CHECK), dtype=np.complex)
    gfactor = get_gfac(i)
    idxplus = (1 + i) % 2 # switch between color diagonal and mixed
    for tdis in range(LT_CHECK): # tdis distance to op
        for gidx, gfac in enumerate(gfactor):
            h_tdis = pieces[(0+idxplus, gidx, tdis)]
            hp_tdis = pieces[(2+idxplus, gidx, tdis)]
            if i in [1, 2]:

                # 1/sqrt(2) * h
                ret[tdis] += h_tdis* 1.0/(math.sqrt(2))

            elif i in [3, 4, 5, 6]:

                # 2Q1-Q1' = [ 2h-h' ] * 1/sqrt(2)
                ret[tdis] += (2*h_tdis-hp_tdis) * 1.0/(math.sqrt(2))

            elif i in [7, 8, 9, 10]: # ew penguin

                # 3/2 [ Q1 (e_d+e_u) - Q1' * e_d ] = [ h + h' ] 1/(2*sqrt(2))
                ret[tdis] += (h_tdis+hp_tdis)*1.0/2.0/math.sqrt(2)

            ret[tdis] *= gfac
    return ret



def qi_proj_sigma_type3(pieces, i):
    """Project onto Q_i (k->sigma), type3"""
    ret = np.zeros((LT_CHECK), dtype=np.complex)
    gfactor = get_gfac(i)
    idxplus = (1 + i) % 2 # switch between color diagonal and mixed
    for tdis in range(LT_CHECK): # tdis distance to op
        for gidx, gfac in enumerate(gfactor):
            i_tdis = pieces[(0+idxplus, gidx, tdis)] # i
            ip_tdis = pieces[(2+idxplus, gidx, tdis)] # i'
            icap_tdis = pieces[(4+idxplus, gidx, tdis)] # I
            icap_p_tdis = pieces[(6+idxplus, gidx, tdis)] # I'

            if i in [1, 2]:

                #  [ i ] *(1)/sqrt(2)
                ret[tdis] += (i_tdis) * (1.0)/(math.sqrt(2))

            elif i in [3, 4, 5, 6]:

                # 2Q1-Q1' + (I-I')*1/sqrt(2) = [ 2i-i'+I-I' ] * 1/sqrt(2)
                ret[tdis] += (2*i_tdis-ip_tdis+icap_tdis-icap_p_tdis) * 1.0/(
                    math.sqrt(2))

            elif i in [7, 8, 9, 10]: # ew penguin

                # 3/2 [ Q1 (e_d+e_u) - e_d*Q1' +(
                # I-I') * e_s * 1/sqrt(2) ] = [
                # i +i' - I + I' ] * 1/(2*sqrt(2))
                ret[tdis] += (i_tdis+ip_tdis-icap_tdis+icap_p_tdis) * 1.0/(
                    math.sqrt(2))

            ret[tdis] *= gfac # spin sign
    return ret


def isospin0_proj_mix3(mix3, opposite=False):
    """Project Mix3 onto isospin 0 pipi
    in other words, generate mix3 subtraction term
    (currently handles only sigma and pipi)
    """
    shape = ()
    for momdiag in mix3:
        shape = mix3[momdiag].shape
        break
    ret = defaultdict(lambda: np.zeros(shape, dtype=np.complex))
    for momdiag in mix3:
        pieces = mix3[momdiag]
        if opposite:
            ret[momdiag][1, :] = pieces[1, :] * (-3.0/math.sqrt(3))
            ret[momdiag][0, :] = pieces[0, :] * (-1.0/math.sqrt(2))
        else:
            ret[momdiag][0, :] = pieces[0, :] * (-3.0/math.sqrt(3))
            ret[momdiag][1, :] = pieces[1, :] * (-1.0/math.sqrt(2))
    return ret

def isospin0_proj_mix4(mix4, opposite=False):
    """Project Mix3 onto isospin 0 pipi
    in other words, generate mix3 subtraction term
    (currently handles only sigma and pipi)
    """
    shape = ()
    for momdiag in mix4:
        shape = mix4[momdiag].shape
        break
    ret = defaultdict(lambda: np.zeros(shape, dtype=np.complex))
    for momdiag in mix4:
        pieces = mix4[momdiag]
        if opposite:
            ret[momdiag][1, :] = pieces[1, :] * (3.0/math.sqrt(3))
            ret[momdiag][0, :] = pieces[0, :] * (2.0/math.sqrt(2))
        else:
            ret[momdiag][0, :] = pieces[0, :] * (3.0/math.sqrt(3))
            ret[momdiag][1, :] = pieces[1, :] * (2.0/math.sqrt(2))
    return ret


def qi_proj_type3(pieces, i, isostr):
    """Project onto Q_i (k->pipi), type3"""
    ret = np.zeros((LT_CHECK), dtype=np.complex)
    gfactor = get_gfac(i)
    idxplus = (1 + i) % 2 # switch between color diagonal and mixed
    for tdis in range(LT_CHECK): # tdis distance to op
        for gidx, gfac in enumerate(gfactor):
            i_tdis = pieces[(0+idxplus, gidx, tdis)] # i
            ip_tdis = pieces[(2+idxplus, gidx, tdis)] # i'
            icap_tdis = pieces[(4+idxplus, gidx, tdis)] # I
            icap_p_tdis = pieces[(6+idxplus, gidx, tdis)] # I'

            if i in [1, 2]:

                if isostr == 'I0': #  [ i ] *(3)/sqrt(3)
                    ret[tdis] += (i_tdis) * (3.0)/(math.sqrt(3))

                elif isostr == 'I2': # [ i ] * 0
                    ret[tdis] += 0

            elif i in [3, 4, 5, 6]:

                if isostr == 'I0': # [2i - i' + I - I'] * 3/sqrt(3)
                    ret[tdis] += (
                        2*i_tdis-ip_tdis+icap_tdis-icap_p_tdis) * 3.0/(
                            math.sqrt(3))

                elif isostr == 'I2': # 2*Q1 = [ i ] * 0
                    ret[tdis] += 0

            elif i in [7, 8, 9, 10]: # ew penguin

                if isostr == 'I0': # 3/2 [ Q1 (e_d+e_u) - (
                    # I'-I) * e_s * 3/sqrt(3)+ i'* e_d * 3/sqrt(3) ] = [
                    # i +i' + - I + I' ] * 3/(2*sqrt(3))
                    ret[tdis] += (
                        i_tdis+ip_tdis-icap_tdis+icap_p_tdis) * 3.0/(
                            2*math.sqrt(3))

                elif isostr == 'I2': # 3/2 Q1 (e_d+e_u) = [i] * 0
                    ret[tdis] += 0

            ret[tdis] *= gfac # spin sign
    return ret



def qi_proj_type4(pieces, i):
    """Project onto Q_i (k->pipi), type4"""
    ret = np.zeros((LT_CHECK), dtype=np.complex)
    gfactor = get_gfac(i)
    idxplus = (1 + i) % 2 # switch between color diagonal and mixed
    for tdis in range(LT_CHECK): # tdis distance to op
        for gidx, gfac in enumerate(gfactor):
            g_tdis = pieces[0+idxplus, gidx, tdis]
            gp_tdis = pieces[2+idxplus, gidx, tdis]
            gcap_tdis = pieces[4+idxplus, gidx, tdis]
            gcap_p_tdis = pieces[6+idxplus, gidx, tdis]
            if i in [1, 2]:

                # g * -3/sqrt(3)
                ret[tdis] += g_tdis * (-3.0)/(math.sqrt(3))

            elif i in [3, 4, 5, 6]:

                # -6 g /sqrt(3) + 3 g'/sqrt(3) - 3G/sqrt(
                # 3) + 3G'/sqrt(3) = [ -2g+g'-G+G' ] * 3/sqrt(3)
                ret[tdis] += (
                    -2*g_tdis + gp_tdis - gcap_tdis + gcap_p_tdis) * 3.0/math.sqrt(
                        3)

            elif i in [7, 8, 9, 10]: # ew penguin

                # 3/2 [ Q1 (e_d+e_u) - (G-G') * e_s * 3/sqrt(
                # 3)+ g'* e_d * 3/sqrt(3) ] = [ -g-g' + G-G'  ] 3/(2*sqrt(3))
                ret[tdis] += (
                    -g_tdis-gp_tdis+gcap_tdis-gcap_p_tdis) * 3.0/2.0/math.sqrt(3)

            ret[tdis] *= gfac
    return ret

def qi_proj_sigma_type4(pieces, i):
    """Project onto Q_i (k->sigma), type4"""
    ret = np.zeros((LT_CHECK), dtype=np.complex)
    gfactor = get_gfac(i)
    idxplus = (1 + i) % 2 # switch between color diagonal and mixed
    for tdis in range(LT_CHECK): # tdis distance to op
        for gidx, gfac in enumerate(gfactor):
            g_tdis = pieces[(0+idxplus, gidx, tdis)]
            gp_tdis = pieces[(2+idxplus, gidx, tdis)]
            gcap_tdis = pieces[(4+idxplus, gidx, tdis)]
            gcap_p_tdis = pieces[(6+idxplus, gidx, tdis)]
            if i in [1, 2]:

                # g * -2/sqrt(2)
                ret[tdis] += g_tdis * (-2.0)/(math.sqrt(2))

            elif i in [3, 4, 5, 6]:

                # 2Q1-Q1'-(G-G')*2/sqrt(2) = [ -2g+g'-G+G' ] * 2/sqrt(2)
                ret[tdis] += (
                    -2*g_tdis+gp_tdis-gcap_tdis+gcap_p_tdis)*2.0/math.sqrt(2)

            elif i in [7, 8, 9, 10]: # ew penguin

                # 3/2 [ Q1 (e_d+e_u) -Q1'*e_d - (
                # G-G') * e_s * 2/sqrt(2)] = [ -g-g' + G-G'  ] * 3/(sqrt(2))
                ret[tdis] += (
                    -g_tdis-gp_tdis+gcap_tdis-gcap_p_tdis) * 3.0/math.sqrt(2)


            ret[tdis] *= gfac
    return ret
