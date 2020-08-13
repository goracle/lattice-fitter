#!/usr/bin/python3

import pickle
import numpy as np

def main():
    """switch between strong and weak cuts (via proc_params.p)"""
    fn1 = pickle.load(open('proc_params.p', 'rb'))
    fn1['strong cuts'] = not fn1['strong cuts']
    print("strong cuts switched to:", fn1['strong cuts'])
    pickle.dump(fn1, open('proc_params.p', 'wb'))


if __name__ == '__main__':
    main()
