import sys
import re
import multiprocessing

def unique_id():
    """Get worker id"""
    ret = multiprocessing.current_process().name
    ret = re.sub('ForkPoolWorker-', '', ret)
    return ret

def f(x):
    """test function"""
    print(unique_id())
    return x * x

if __name__ == '__main__':
    p = multiprocessing.Pool()
    print(p.map(f, range(6)))
    sys.exit()
