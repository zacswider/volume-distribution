import os
from tqdm import tqdm
from skimage import filters
from tifffile import imread, imwrite
import multiprocess as mp

def main():
    processing_dir = '/Volumes/bigData/wholeMount_volDist/220726-0805_Fix_Emb_Flvw_Chn1GAP_PI_aTub647_Processed/0_N2V_Denoised/16bit_scaleZ'
    save_dir = os.path.join(processing_dir, 'sbdl2_processed')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    im_names = [f for f in os.listdir(processing_dir) if f.endswith('.tif') and not f.startswith('.')]

    def blur_image(im_name):
        im_base = im_name.split('.')[0]
        im = imread(os.path.join(processing_dir, im_name))
        sbdl2 = filters.difference_of_gaussians(im, low_sigma=2, high_sigma=128)
        imwrite(os.path.join(save_dir, f'{im_base}_sbdl2.tif'), sbdl2)
    
    with mp.Pool() as pool:
        r = list(tqdm(pool.imap(blur_image, im_names), total=len(im_names)))

if __name__ == '__main__':
    main()