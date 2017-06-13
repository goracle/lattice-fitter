#!/usr/bin/python

from math import sqrt
from .sum_blks import sum_blks
from . import read_file as rf
import os

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
        

A_1plus=[(1/sqrt(6),'pipi',[[1,0,0],[-1,0,0]]),(1/sqrt(6),'pipi',[[0,1,0],[0,-1,0]]),(1/sqrt(6),'pipi',[[0,0,1],[0,0,-1]]),(1/sqrt(6),'pipi',[[-1,0,0],[1,0,0]]),(1/sqrt(6),'pipi',[[0,-1,0],[0,1,0]]),(1/sqrt(6),'pipi',[[0,0,-1],[0,0,1]])]

T_1_1minus=[(-1/2,'pipi',[[1,0,0],[-1,0,0]]),(complex(0,-1/2),'pipi',[[0,1,0],[0,-1,0]]),(1/2,'pipi',[[-1,0,0],[1,0,0]]),(complex(0,-1/2),'pipi',[[0,-1,0],[0,1,0]])]

T_1_3minus=[(1/2,'pipi',[[1,0,0],[-1,0,0]]),(complex(0,-1/2),'pipi',[[0,1,0],[0,-1,0]]),(-1/2,'pipi',[[-1,0,0],[1,0,0]]),(complex(0,1/2),'pipi',[[0,-1,0],[0,1,0]])]

T_1_2minus=[(1/sqrt(2),'pipi',[[0,0,1],[0,0,-1]]),(-1/sqrt(2),'pipi',[[0,0,-1],[0,0,1]])]

oplist={'A_1plus':A_1plus,'T_1_1minus':T_1_1minus,'T_1_3minus':T_1_3minus,'T_1_2minus':T_1_2minus}
partlist=set([])
for op in oplist:
    for item in oplist[op]:
        partlist.add(item[1])

def partstr(srcpart,snkpart):
    if srcpart == snkpart and srcpart == 'pipi':
        particles = 'pipi'
    else:
        particles = snkpart+srcpart
    return particles

def op_list():
    for srcpart in partlist:
        for snkpart in partlist:
            partchk=partstr(srcpart,snkpart)
            for op in oplist:
                print("Doing op:",op,"for particles:",partchk)
                coeffs_arr = []
                for src in oplist[op]:
                    for snk in oplist[op]:
                        if src[1] == 'pipi' and snk[1] == 'pipi':
                            partS='pipi'
                        else:
                            partS=snk[1]+src[1]
                        if partS != partchk:
                            continue
                        coeff=src[0]*snk[0]
                        pstr=momstr(src[2],snk[2])
                        d = partS+"_"+pstr
                        if not os.path.isdir(d):
                            print("dir",d,"is missing")
                            sys.exit(1)
                        coeffs_arr.append((d,coeff))
                outdir=partchk+"_"+op
                sum_blks(outdir,coeffs_arr)
    print("End of operator list.")
    return
            








def main():
    op_list()


























if __name__ == "__main__":
    main()

