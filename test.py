import argparse
import numpy as np

if __name__ == '__main__':
    a = set()
    a.update([1, 2])
    a.update([0, 3])
    a.update([3, 3])
    print(a)