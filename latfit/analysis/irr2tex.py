"""Convert Irrep name to LaTeX"""

def irr2tex(irr):
    """Convert irrep name to latex"""
    irr = irr.split('mom')[0]
    sup = None
    sub = None
    if 'MINUS' in irr:
        sup = '-'
    elif 'PLUS' in irr:
        sup = '+'
    if 'A' in irr:
        name = 'A'
    elif 'B' in irr:
        name = 'B'
    elif 'T' in irr:
        name = 'T'
    digits = [str(i) for i in range(10)]
    for i in digits:
        if i in irr:
            sub = i
    ret = ' $'+name
    if sup is not None:
        ret += '^{'+sup+'}'
    if sub is not None:
        ret += '_{'+sub+'}'
    ret += '$, '
    return ret
