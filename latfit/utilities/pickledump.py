#!/usr/bin/python3
"""dump pickle file"""

import sys
import pickle


def main():
    """pickle dump main"""

    for fil in sys.argv[1:]:
        fn1 = open(fil, 'rb')
        try:
            fn1 = pickle.load(fn1)
        except KeyError:
            print("KeyError:", fil, "may not be a pickle file; corrupt.")
            continue
        print(fil, "contents:", fn1)



if __name__ == '__main__':
    main()
