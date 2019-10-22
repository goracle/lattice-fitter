"""Give the rho (vector meson) operator
 the proper polarization
(due to group theory from A. Meyer)
"""
import sys
import numpy as np

def lstr(arr):
    """Make sure the string will
    convert back to a list"""
    ret = list(arr)
    ret = str(ret)
    return ret

def row(irrf):
    """Find row number of irrep from string"""
    ret = irrf.split('_')[1]
    return int(ret)


def rho_pol(irr, mom):
    """Find rho polarization coefficients"""
    #pols for p1
    key = ""
    if 'A_1PLUS' in irr and sum(np.abs(mom)) == 1:
        key += lstr(p1_a1p(mom))
    elif 'B' in irr and sum(np.abs(mom)) == 1:
        add = p1_b0(mom)
        if row(irr):
            add = np.conj(add)
        add = list(add)
        key += lstr(add)

    #pols for p11
    elif 'A_1PLUS' in irr and sum(np.abs(mom)) == 2:
        key += lstr(p11_a1p(mom))
    elif 'A_2PLUS' in irr and sum(np.abs(mom)) == 2:
        key += lstr(p11_a2p(mom))
    elif 'A_2MINUS' in irr and sum(np.abs(mom)) == 2:
        key += lstr(p11_a2m(mom))

    #pols for p111
    elif 'A_1PLUS' in irr and sum(np.abs(mom)) == 3:
        key += lstr(p111_a1p(mom))
    elif 'B' in irr and sum(np.abs(mom)) == 3:
        if row(irr):
            key += lstr(p111_b1(mom))
        else:
            key += lstr(p111_b0(mom))
    return key

def p111_b1(mom):
    """Get the B row 1  pol vector for p111"""
    aone = 1/np.sqrt(6)
    atwo = 1/2/np.sqrt(6)
    athree = 1j/2/np.sqrt(2)
    mom = list(mom)
    if mom == [-1, -1, -1]:
        ret = [atwo-athree, -1*aone, atwo+athree]
    elif mom == [-1, -1, 1]:
        ret = [atwo-athree, atwo+athree, aone]
    elif mom == [-1, 1, -1]:
        ret = [atwo-athree, -1*atwo-athree, -1*aone]
    elif mom == [-1, 1, 1]:
        ret = [atwo-athree, aone, -1*atwo-athree]
    elif mom == [1, -1, -1]:
        ret = [-1*atwo+athree, atwo+athree, -1*aone]
    elif mom == [1, -1, 1]: #
        ret = [aone, atwo+athree, -1*atwo+athree]
    elif mom == [1, 1, -1]: #
        ret = [-1*atwo-athree, -1*atwo+athree, -1*aone]
    elif mom == [1, 1, 1]: #
        ret = [aone, -1*atwo+athree, -1*atwo-athree]
    else:
        mdie(mom)
    return ret



def p111_b0(mom):
    """Get the B row 0  pol vector for p111"""
    aone = 1/np.sqrt(6)
    atwo = 1/2/np.sqrt(6)
    athree = 1j/2/np.sqrt(2)
    mom = list(mom)
    if mom == [-1, -1, -1]:
        ret = [-1*aone, atwo-athree, atwo+athree]
    elif mom == [-1, -1, 1]:
        ret = [-1*aone, atwo+athree, -1*atwo+athree]
    elif mom == [-1, 1, -1]:
        ret = [-1*aone, -1*atwo-athree, atwo-athree]
    elif mom == [-1, 1, 1]:
        ret = [-1*aone, -1*atwo+athree, -1*atwo-athree]
    elif mom == [1, -1, -1]:
        ret = [aone, atwo+athree, atwo-athree]
    elif mom == [1, -1, 1]: #
        ret = [-1*atwo+athree, atwo+athree, aone]
    elif mom == [1, 1, -1]: #
        ret = [-1*atwo-athree, aone, atwo-athree]
    elif mom == [1, 1, 1]: #
        ret = [-1*atwo+athree, aone, -1*atwo-athree]
    else:
        mdie(mom)
    return ret

def p1_b0(mom):
    """Get the B row 0  pol vector for p1"""
    mom = list(mom)
    if mom == [0, 0, -1]:
        ret = [0.5, 0.5j, 0]
    elif mom == [0, -1, 0]:
        ret = [0.5, 0, -0.5j]
    elif mom == [-1, 0, 0]:
        ret = [0, -0.5j, 0.5]
    elif mom == [1, 0, 0]:
        ret = [0, -0.5j, -0.5]
    elif mom == [0, 1, 0]:
        ret = [0.5, 0, 0.5j]
    elif mom == [0, 0, 1]:
        ret = [0.5, -0.5j, 0]
    else:
        mdie(mom)
    return ret
    


def p111_a1p(mom):
    """Get the A_1^+ pol vector for p111"""
    ret = [np.sign(i)/np.sqrt(6) for i in mom]
    assert len(ret) == 3
    return ret


def p11_a1p(mom):
    """Get the A_1^+ vector"""
    ret = [0.5*np.sign(i) for i in mom]
    assert len(ret) == 3
    return ret

def p11_a2m(mom):
    """Get the A_2^- vector
    is 1/sqrt(2)
    if the mom entry is 0
    and is 0 otherwise
    """
    ret = [1/np.sqrt(2)*(
        1-np.abs(np.sign(i))) for i in mom]
    assert len(ret) == 3
    return ret

def p1_a1p(mom):
    """p1, A_1^+ pol vec"""
    ret = [np.sign(i)/np.sqrt(2) for i in mom]
    assert len(ret) == 3
    return ret

def p11_a2p(mom):
    """Get A_2^+
    No clear pattern, so exhaustive list
    """
    mom = list(mom)

    # all neg, cyclic
    if mom == [-1, -1, 0]:
        ret = [-0.5, 0.5, 0] #
    elif mom == [-1, 0, -1]:
        ret = [0.5, 0, -0.5] #
    elif mom == [0, -1, -1]:
        ret = [0, -0.5, 0.5] #

    # one neg
    elif mom == [-1, 0, 1]:
        ret = [0.5, 0, 0.5]
    elif mom == [0, 1, -1]:
        ret = [0, -0.5, -0.5]
    elif mom == [1, -1, 0]:
        ret = [-0.5, -0.5, 0]
    elif mom == [-1, 1, 0]:
        ret = [0.5, 0.5, 0]
    elif mom == [0, -1, 1]:
        ret = [0, -0.5, -0.5]
    elif mom == [1, 0, -1]:
        ret = [0.5, 0, 0.5]

    # all pos, swap
    elif mom == [1, 1, 0]:
        ret = [0.5, -0.5, 0]
    elif mom == [1, 0, 1]:
        ret = [0.5, 0, -0.5]
    elif mom == [0, 1, 1]:
        ret = [0, -0.5, 0.5]
    else:
        mdie(mom)
    a2p_pos_check(ret, mom)
    a2p_same_sign_check(ret, mom)

    return ret

def mdie(mom):
    """Throw error, print mom"""
    print("bad mom spec", mom)
    assert None

def a2p_pos_check(ret, mom):
     """Check to make sure the two vectors
     are 0 in the same place"""
     for i, j in zip(ret, mom):
         iandj = i and j
         niandnj = not i and not j
         assert iandj or niandnj,\
             str(ret)+" "+str(mom)

def a2p_same_sign_check(ret, mom):
    """If the momentum components have the same sign
    the pol vector components should not and vice-versa
    """
    same = np.sum(mom)
    rsame = np.sum(ret)
    try:
        if rsame:
            assert np.abs(rsame) == 1
        if same:
            assert not rsame
        else:
            assert rsame
    except AssertionError:
        print(ret)
        print(mom)
        raise

