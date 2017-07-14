#!/usr/bin/python3

from math import sqrt
from sum_blks import sum_blks
import read_file as rf
import os
import sys
import re

def momstr(psrc,psnk):
    pipi = ''
    if len(psrc) == 2 and len(psnk) == 2:
        vertices = 4
    elif len(psrc) == 3 and type(psrc[0]) is int and len(psnk) == 2:
        vertices = 3
        pipi = 'snk'
    elif len(psnk) == 3 and type(psnk[0]) is int and len(psrc) == 2:
        vertices = 3
        pipi = 'src'
    elif len(psnk) == len(psrc) and len(psrc) == 3 and type(psrc[0]) is int:
        vertices = 2
    else:
        pstr = None
    if vertices == 4:
        pstr = 'mom1src'+rf.ptostr(psrc[0])+'_mom2src'+rf.ptostr(psrc[1])+'_mom1snk'+rf.ptostr(psnk[0])
    elif vertices == 3 and pipi == 'src':
        pstr = 'momsrc'+rf.ptostr(psrc[0])+'_momsnk'+rf.ptostr(psnk)
    elif vertices == 3 and pipi == 'snk':
        pstr = 'momsrc'+rf.ptostr(psrc)+'_momsnk'+rf.ptostr(psnk[0])
    elif vertices == 2:
        if psrc[0] != psnk[0] or psrc[1] != psnk[1] or psrc[2] != psnk[2]:
            pstr = None
        else:
            pstr = 'mom'+rf.ptostr(psrc)
    return pstr
        

A_1plus=[
    (1/sqrt(6),'pipi',[[1,0,0],[-1,0,0]]),
    (1/sqrt(6),'pipi',[[0,1,0],[0,-1,0]]),
    (1/sqrt(6),'pipi',[[0,0,1],[0,0,-1]]),
    (1/sqrt(6),'pipi',[[-1,0,0],[1,0,0]]),
    (1/sqrt(6),'pipi',[[0,-1,0],[0,1,0]]),
    (1/sqrt(6),'pipi',[[0,0,-1],[0,0,1]]),
    (1,'S_pipi',[[0,0,0],[0,0,0]]),
#    (1,'sigma',[0,0,0]),
    (1/sqrt(12),'UUpipi',[[0,1,1],[0,-1,-1]]),
    (1/sqrt(12),'UUpipi',[[1,0,1],[-1,0,-1]]),
    (1/sqrt(12),'UUpipi',[[1,1,0],[-1,-1,0]]),
    (1/sqrt(12),'UUpipi',[[0,1,1],[0,-1,-1]]),
    (1/sqrt(12),'UUpipi',[[1,0,1],[-1,0,-1]]),
    (1/sqrt(12),'UUpipi',[[1,1,0],[-1,-1,0]]),
    (1/sqrt(12),'UUpipi',[[0,-1,1],[0,1,-1]]),
    (1/sqrt(12),'UUpipi',[[-1,0,1],[1,0,-1]]),
    (1/sqrt(12),'UUpipi',[[-1,1,0],[1,-1,0]]),
    (1/sqrt(12),'UUpipi',[[0,-1,1],[0,1,-1]]),
    (1/sqrt(12),'UUpipi',[[-1,0,1],[1,0,-1]]),
    (1/sqrt(12),'UUpipi',[[-1,1,0],[1,-1,0]]),
]

#A_1plus_sigma=[(1,'sigma',[0,0,0])]

T_1_1minus=[(-1/2,'pipi',[[1,0,0],[-1,0,0]]),(complex(0,-1/2),'pipi',[[0,1,0],[0,-1,0]]),(1/2,'pipi',[[-1,0,0],[1,0,0]]),(complex(0,-1/2),'pipi',[[0,-1,0],[0,1,0]])]

T_1_3minus=[(1/2,'pipi',[[1,0,0],[-1,0,0]]),(complex(0,-1/2),'pipi',[[0,1,0],[0,-1,0]]),(-1/2,'pipi',[[-1,0,0],[1,0,0]]),(complex(0,1/2),'pipi',[[0,-1,0],[0,1,0]])]

T_1_2minus=[(1/sqrt(2),'pipi',[[0,0,1],[0,0,-1]]),(-1/sqrt(2),'pipi',[[0,0,-1],[0,0,1]])]

oplist={'A_1plus':A_1plus,'T_1_1minus':T_1_1minus,'T_1_3minus':T_1_3minus,'T_1_2minus':T_1_2minus}
part_list=set([])
for op in oplist:
    for item in oplist[op]:
        part_list.add(item[1])

def partstr(srcpart,snkpart):
    if srcpart == snkpart and srcpart == 'pipi':
        particles = 'pipi'
    else:
        particles = snkpart+srcpart
    return particles

part_combs=set([])
for src in part_list:
    for snk in part_list:
        part_combs.add(partstr(src,snk))

def op_list():
    for op in oplist:
        coeffs_tuple = []
        for src in oplist[op]:
            for snk in oplist[op]:
                part_str=partstr(src[1],snk[1])
                coeff=src[0]*snk[0]
                p_str=momstr(src[2],snk[2])
                d = part_str+"_"+p_str
                d=re.sub('S_','',d)
                d=re.sub('UU','',d)
                d=re.sub('pipipipi','pipi',d)
                if not os.path.isdir(d):
                    if not os.path.isdir('sep4/'+d):
                        print("For op:", op)
                        print("dir",d,"is missing")
                        sys.exit(1)
                    else:
                        d='sep4/'+d
                coeffs_tuple.append((d,coeff,part_str))
        coeffs_arr=[]
        print("trying",op)
        for parts in part_combs:
            outdir=parts+"_"+op
            coeffs_arr=[(tup[0],tup[1]) for tup in coeffs_tuple if tup[2] == parts]
            if not coeffs_arr:
                continue
            print("Doing",op,"for particles",parts)
            sum_blks(outdir,coeffs_arr)
    print("End of operator list.")
    return

def main():
    op_list()


if __name__ == "__main__":
    main()

