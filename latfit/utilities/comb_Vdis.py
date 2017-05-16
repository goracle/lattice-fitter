#!/usr/bin/python

from read_file import *
import os.path

def procV(fn1,fn2):
    print "processing:",fn1,fn2
    data = comb_dis(fn1,fn2)
    data_out = np.zeros((len(data),len(data[0])),dtype=np.object)
    for i in range(len(data)):
        for j in range(len(data[0])):
            y=str(data[i][j].real)+" "+str(data[i][j].imag)
            data_out[i][j]=y
    outf = "traj_"+str(traj(fn1))+"_FigureV_sep"+str(sep(fn1))
    if mom(fn1) and mom(fn2):
        p1src,p2src=mom(fn1)
        p2snk,p1snk=mom(fn2) #reverse since at the sink
        outf+="_mom1src"+ptostr(p1src)+"_mom2src"+ptostr(p2src)+"_mom1snk"+ptostr(p1snk)
        if(os.path.isfile(outf)):
            print "Skipping:", outf
            print "File exists."
            return
        write_mat_str(data_out,outf)
    return

def main():
    onlyfiles=[f for f in listdir('.') if isfile(join('.',f))]
    for fn1 in onlyfiles:
        for fn2 in onlyfiles:
            if traj(fn1) == traj(fn2) and sep(fn1) == sep(fn2):
                if figure(fn1) == 'Vdis' and figure(fn2) == 'Vdis':
                    procV(fn1,fn2)
    return

if __name__ == "__main__":
    main()
