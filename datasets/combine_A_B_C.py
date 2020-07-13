"""
This file simply combines the output from R2HCycle with the corresponding ground truth for training/testing H2RGAN in a paired way.
A - HSI Image
B - Clean RGB/Ground truth
C - Hazy RGB (Not required for training, but can be used for visualization. Feel free to skip by commenting out lines and passing None in C. 

It might be required to run this file for each dataset separately, due to the naming conventions changing with dataset. 
To do so, please change the following lines according to your need. 
1. Path to image A,B,C
2. The way image B and C are found (line 64 onwards)
3. The destination folder
"""

import os
import numpy as np
import cv2
import argparse
import h5py

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A, hazy HSI', type=str, default='/data1/final_results/RESIDE/SOTS/indoor/hazy/4hsi_vanilla_version2/test_latest/') # a folder up
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B, clean RGB', type=str, default='/data1/Datasets/RESIDE/SOTS/SOTS/indoor/gt/') #exact 
parser.add_argument('--fold_C', dest='fold_C', help='input directory for image C, hazy RGB', type=str, default='/data1/Datasets/RESIDE/SOTS/SOTS/indoor/hazy/') #exact

parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='../datasets/H2RGAN/')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)' ,action='store_true')
parser.add_argument('--typ', dest='typ', type = str, default = 'test')
                   
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

splits = os.listdir(args.fold_A) #can be helpful is naming convention is same and all images are in one root folder. 

for sp in splits:
    if sp == 'images':

        img_fold_A = os.path.join(args.fold_A, sp)
        img_fold_B = args.fold_B
        img_fold_C = args.fold_C
        
        img_list = os.listdir(img_fold_A)

        if args.use_AB:
            img_list = [img_path for img_path in img_list if '_A.' in img_path]

        num_imgs = min(args.num_imgs, len(img_list))
        print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
        img_fold_AB = os.path.join(args.fold_AB)
        if not os.path.isdir(img_fold_AB):
            os.makedirs(img_fold_AB)
        print('split = %s, number of images = %d' % (sp, num_imgs))
        for n in range(num_imgs):
            name_A = img_list[n]
            if name_A.endswith('_fake.hdf5'):
                path_A = os.path.join(img_fold_A, name_A)
            else:
                continue

            if args.use_AB:
                name_B = name_A.replace('_A.', '_B.')
            else: # This has to be customized according to the names your files have
                x = name_A.split('_')
                name_B = x[0]+'.png'
                name_C = x[0]+'_'+x[1]+'.png'
                path_B = os.path.join(img_fold_B, name_B)
                path_C = os.path.join(img_fold_C, name_C)
                

            if os.path.isfile(path_A) and os.path.isfile(path_B) and os.path.isfile(path_C):

                name_AB = name_A
                if args.use_AB:
                    name_AB = name_AB.replace('_A.', '.')  # remove _A
                path_AB = os.path.join(img_fold_AB, 'sots_'+name_AB)

                with h5py.File(path_A, 'r') as f:
                    dA = f.get('data').value
                    data_A = dA

                data_B = cv2.imread(path_B, 1) 
                data_C = cv2.imread(path_C, 1)

                hf = h5py.File(path_AB, 'w')
                hf.create_dataset('hazy_hsi', data=data_A)
                hf.create_dataset('clean_rgb', data=data_B)
                hf.create_dataset('hazy_rgb',data = data_C)
                hf.close()
                print(n)

