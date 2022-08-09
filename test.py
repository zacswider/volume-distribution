from voldist_tools.vol_utils import EmbryoSeg
from pathlib import Path
import numpy as np
import os
import time
import multiprocess as mp
from tqdm import tqdm


pool = mp.Pool()

mylist = list(range(10))

def return_random(listnum):
    return f'{listnum} returned {np.random.randint(10,20)}'

results = pool.map(return_random, mylist)
print(results)


'''
# Linear vs multiprocessing
if __name__ == "__main__":

    analysis_dir = '/Volumes/bigData/wholeMount_volDist/220805-0712_Combined'

    subdirs = [os.path.join(analysis_dir, p) for p in os.listdir(analysis_dir) if os.path.isdir(os.path.join(analysis_dir, p)) and not p.startswith('.')]

    start = time.time()
    for s in tqdm(subdirs):
        emb = EmbryoSeg(s)
        emb.find_emb_centroid()
        emb.filter_cell_labels()
        emb.filter_spindle_labels()
    end = time.time()
    print(f'linear calculations finished in {end - start} seconds')

    start = time.time()

    def make_class(mystring):
        emb = EmbryoSeg(mystring)
        emb.find_emb_centroid(overwrite = True)
        emb.filter_cell_labels(overwrite = True)
        emb.filter_spindle_labels(overwrite = True)

    with mp.Pool() as pool:
        pool.map(make_class, subdirs)

    end = time.time()
    print(f'multiprocessing finished in {end - start} seconds')
'''