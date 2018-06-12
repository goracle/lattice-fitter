"""Get directories which have generalized eigenvalue matrix elements"""
import sys
import re
import latfit.utilities.read_file as rf

def gevp_dirs(isospin, momstr, irrep, dim, sigma=False):
    """Return array of gevp dirs"""
    dirstr = 'I'+str(isospin)+"/"
    irrepstr = '_'+str(irrep)
    endstr = '.jkdat' if irrep != 'A1' else comp(momstr)+'.jkdat'
    retdirs = [[None for i in range(dim)] for j in range(dim)]
    for i in range(dim):
        for j in range(dim):
            istr = hierarchy(i, dim, isospin, sigma)
            jstr = hierarchy(j, dim, isospin, sigma)
            if istr == 'pipi' and jstr == 'pipi':
                jstr = ''
            retdirs[i][j] = dirstr+istr+jstr+irrepstr+endstr
    return retdirs

def hierarchy(index, dim, isospin, sigma):
    """Return the operator for this dimension, in ascending energy order"""
    if isospin == 0 or isospin == 2:
        if index == 0:
            retstr = 'S_pipi'
        elif index > 0:
            if dim == 2:
                if sigma:
                    retstr = 'sigma'
                else:
                    retstr = 'pipi'
            elif dim == 3:
                if isospin == 0:
                    if index == 1:
                        retstr = 'sigma'
                    else:
                        retstr = 'pipi'
                else:
                    if index == 1:
                        retstr = 'pipi'
                    elif index == 2:
                        retstr = 'UUpipi'
            elif dim == 4:
                assert isospin == 0, "4x4 I=2 not supported"
                if index == 1:
                    retstr = 'sigma'
                elif index == 2:
                    retstr = 'pipi'
                elif index == 3:
                    retstr = 'UUpipi'
    else: # isospin = 1
        if index == 0:
            retstr = 'pipi'
        elif index == 1:
            retstr = 'rho'
        else:
            print(">2x2 GEVP I=1 not yet supported.")
            raise
    return retstr

def comp(momstr):
    """Get string for A1 irrep."""
    compstr = re.sub('momtotal', '', momstr)
    comp = rf.procmom(compstr)
    if comp[0]**2+comp[1]**2+comp[2]**2 > 1:
        assert 0, "Center of mass > 1 unit not supported."
    retstr = ''
    for j, i in enumerate(comp):
        if i < 0:
            retstr = 'm'
        if i**2 > 0:
            if j == 0:
                retstr = retstr+'x'
            elif j == 1:
                retstr = retstr+'y'
            else:
                retstr = retstr+'z'
    return retstr


# spelled out, old style, for reference only

isospin = 0
sigma = False
momstr = 'momtotal000'

if isospin == 0:
    if not sigma and momstr == 'momtotal000':
        # no sigma
        retdirs = [
            ['I0/S_pipiS_pipi_A_1PLUS.jkdat', 'I0/S_pipipipi_A_1PLUS.jkdat'],
            ['I0/pipiS_pipi_A_1PLUS.jkdat', 'I0/pipi_A_1PLUS.jkdat']
        ]
    elif not sigma and momstr == 'momtotal001':
        retdirs = [
            ['I0/S_pipiS_pipi_A1z.jkdat', 'I0/S_pipipipi_A1z.jkdat'],
            ['I0/pipiS_pipi_A1z.jkdat', 'I0/pipi_A1z.jkdat']
        ]
    elif momstr == 'momtotal001':
        if DIM == 2:
            retdirs = [
                ['I0/S_pipiS_pipi_A1z.jkdat', 'I0/S_pipipipi_A1z.jkdat'],
                ['I0/sigmaS_pipi_A1z.jkdat', 'I0/pipi_A1z.jkdat']
            ]
        
    elif momstr == 'momtotal000':

        if DIM == 2:
            # sigma
            retdirs = [
                ['I0/S_pipiS_pipi_A_1PLUS.jkdat',
                 'I0/S_pipisigma_A_1PLUS.jkdat'],
                ['I0/sigmaS_pipi_A_1PLUS.jkdat',
                 'I0/sigmasigma_A_1PLUS.jkdat']
            ]
        elif DIM == 3:
            # 3x3, I0
            retdirs = [
                ['I0/S_pipiS_pipi_A_1PLUS.jkdat',
                'I0/S_pipisigma_A_1PLUS.jkdat',
                'I0/S_pipipipi_A_1PLUS.jkdat'],
                ['I0/sigmaS_pipi_A_1PLUS.jkdat',
                'I0/sigmasigma_A_1PLUS.jkdat',
                'I0/sigmapipi_A_1PLUS.jkdat'],
                ['I0/pipiS_pipi_A_1PLUS.jkdat',
                'I0/pipisigma_A_1PLUS.jkdat',
                'I0/pipi_A_1PLUS.jkdat']
            ]

        elif DIM == 4:
            # 3x3, I0
            retdirs = [
                ['I0/S_pipiS_pipi_A_1PLUS.jkdat',
                'I0/S_pipisigma_A_1PLUS.jkdat',
                'I0/S_pipipipi_A_1PLUS.jkdat',
                'I0/S_pipiUUpipi_A_1PLUS.jkdat'
                 ]
                ['I0/sigmaS_pipi_A_1PLUS.jkdat',
                'I0/sigmasigma_A_1PLUS.jkdat',
                 'I0/sigmapipi_A_1PLUS.jkdat',
                 'I0/sigmaUUpipi_A_1PLUS.jkdat'
                ],
                ['I0/pipiS_pipi_A_1PLUS.jkdat',
                'I0/pipisigma_A_1PLUS.jkdat',
                 'I0/pipi_A_1PLUS.jkdat',
                 'I0/pipiUUpipi_A_1PLUS.jkdat'
                ],
                [
                 'I0/UUpipiS_pipi_A_1PLUS.jkdat',
                'I0/UUpipisigma_A_1PLUS.jkdat',
                 'I0/UUpipipipi_A_1PLUS.jkdat',
                 'I0/UUpipiUUpipi_A_1PLUS.jkdat'
                ]
            ]


        ##non-zero center of mass momentum, one stationary pion
        # sigma
    else:
        retdirs = [
            ['I0/pipi_A2.jkdat',
             'I0/pipisigma_A2.jkdat'],
            ['I0/sigmapipi_A2.jkdat',
             'I0/sigmasigma_A2.jkdat']
        ]


elif isospin == 2:
    if momstr == 'momtotal000':
        # pipi with one unit of momentum
        retdirs = [
            ['I2/S_pipiS_pipi_A_1PLUS.jkdat', 'I2/S_pipipipi_A_1PLUS.jkdat'],
            ['I2/pipiS_pipi_A_1PLUS.jkdat', 'I2/pipi_A_1PLUS.jkdat']
        ]
    else:
        ##non-zero center of mass momentum, one stationary pion
        # sigma
        retdirs = [
            ['I0/pipi_A2.jkdat',
                'I0/pipisigma_A2.jkdat'],
            ['I0/sigmapipi_A2.jkdat',
                'I0/sigmasigma_A2.jkdat']
        ]

elif isospin == 1:
    retdirs = [
        ['I1/pipi_A_1PLUS.jkdat',
            'I1/pipirho_A_1PLUS.jkdat'],
        ['I1/rhopipi_A_1PLUS.jkdat',
            'I1/rhorho_A_1PLUS.jkdat']
    ]

# 2x2 I = 0
# GEVP_DIRS = [['sep4/pipi_mom1src000_mom2src000_mom1snk000',
# 'sep4/pipisigma_momsrc000_momsnk000'],
# ['sep4/sigmapipi_momsrc000_momsnk000', 'sigmasigma_mom000']]

# GEVP_DIRS = [['sep4/pipi_mom1src000_mom2src000_mom1snk000',
# 'S_pipipipi_A_1PLUS'], ['pipiS_pipi_A_1PLUS', 'pipi_A_1PLUS']]

# 3x3, I2, pipi, 000, 100, 110
# GEVP_DIRS = [['S_pipiS_pipi_A_1PLUS', 'S_pipipipi_A_1PLUS',
# 'S_pipiUUpipi_A_1PLUS'],
# ['pipiS_pipi_A_1PLUS', 'pipi_A_1PLUS', 'pipiUUpipi_A_1PLUS'],
# ['UUpipiS_pipi_A_1PLUS', 'UUpipipipi_A_1PLUS', 'UUpipiUUpipi_A_1PLUS']]

