from voldist_tools.vol_utils import EmbryoSeg
import os
import multiprocess as mp
import time

'''
This script iterates through a folder expecting each subdirectory to contain a set of images:
1) a cellpose segmentation file (.npy)
2) propidium iodide staining file (.tif)
3) anti-tubulin staining file (.tif)
4) DoG-filtered anti-tubulin staining file (.tif)

This script will then process each embryo in parallel and save the results.
'''

if __name__ == '__main__':
    maindir = '/Volumes/bigData/wholeMount_volDist/220805-0712_Combined'
    subdirs = [os.path.join(maindir, d) for d in os.listdir(maindir) if os.path.isdir(os.path.join(maindir, d))]
    
    def process_embryo(embryo):
        e = EmbryoSeg(embryo)
        e.calculate_properties(overwrite=True)
    
    start = time.time()
    with mp.Pool() as pool:
        pool.map(process_embryo, subdirs)
        pool.close()
        pool.join()
        
    end = time.time()
    print(f'Time elapsed: {end - start}')    