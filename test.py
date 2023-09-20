import argparse
import numpy as np

if __name__ == '__main__':
    break_points = list(np.arange(0, 526, 2000))+ [256]
    chunk_info = {ww: False for ww in list(range(len(break_points) - 1))}
    print(chunk_info)