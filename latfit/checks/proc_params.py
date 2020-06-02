"""Get processing params"""
import sys
import pickle
import numpy as np

def look_for_file():
    """Look for the proc_params.p file"""
    fn1 = None
    try:
        fn1 = open('proc_params.p', 'rb')
    except FileNotFoundError:
        print("process params file not found")
    return fn1

def array_from_file(fn1):
    """Get the pickled array"""
    ret = pickle.load(fn1)
    ret = np.asarray(ret)
    return ret

def check_params(matsub, prgrdonly, pionratio, strongcuts):
    """Check config file vs. process param file found in the working dir"""
    fn1 = look_for_file()
    if fn1 is not None:
        arr = array_from_file(fn1)
        assert arr[0] == matsub, (arr, matsub)
        assert arr[1] == prgrdonly, (arr, prgrdonly)
        assert arr[2] == pionratio, (arr, pionratio)
        assert arr[3] == strongcuts, (arr, strongcut)
    else:
        arr = [matsub, prgrdonly, pionratio, strongcuts]
        arr = np.asarray(arr)
        fn1 = open('proc_params.p', 'wb')
        print("writing process param file to lock this directory's config.")
        print("writing: proc_params.p")
        pickle.dump(arr, fn1)

def main():
    """Accept user input to set up the directory"""
    arr = sys.argv[1:]
    if len(sys.argv[1:]) != 5:
        print("input process params (as 0: False or 1: True):")
        print("MATRIX_SUBTRACTION, PR_GROUND_ONLY,",
              "PIONRATIO, STRONGCUTS")
    else:
        arr = [bool(int(i)) for i in arr]
        check_params(*arr)


if __name__ == '__main__':
    main()
