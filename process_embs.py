from voldist_tools.vol_utils import EmbryoSeg
import os
import multiprocess as mp
import time

if __name__ == '__main__':
    maindir = '/Volumes/bigData/wholeMount_volDist/220805-0712_Combined'
    subdirs = [os.path.join(maindir, d) for d in os.listdir(maindir) if os.path.isdir(os.path.join(maindir, d))]
    
    for subdir in subdirs:
        print(f'creating embryo segs for {subdir}')
        e = EmbryoSeg(subdir)
        e.manual_annotation()
    '''
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
    '''
    