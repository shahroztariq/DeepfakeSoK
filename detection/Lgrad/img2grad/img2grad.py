# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '1'
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import sys
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
random_seed = 1000


def main(args):
    # Initialize TensorFlow.
    tflib.init_tf()
    num_gpus       = int(args[1]) if len(args)>1 else 1
    input_dir      = args[2]      if len(args)>2 else './genimg'
    out_dir        = args[3]      if len(args)>3 else './genimg_grad'
    pkl_path       = args[4]      if len(args)>4 else 'networks/karras2019stylegan-bedrooms-256x256.pkl'
    minibatch_size = int(args[5]) if len(args)>5 else 1
    os.makedirs(out_dir, mode=0o777, exist_ok=True)
    nowpath = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(nowpath, pkl_path), 'rb') as f:
        _G, _D, Gs = pickle.load(f)
        
    print(f"[INPUT DIR] ---------- {input_dir}")
    print(f"[OUTPUT DIR] --------- {out_dir}")
    print(f"[MINI BATCH] --------- {minibatch_size}")
    #print(f"[picke _D] --------- {_D}")
    #print(dir)
    _D.run_img2grad(input_dir, out_dir, minibatch_size=minibatch_size, num_gpus = num_gpus)


from glob import glob
if __name__ == "__main__":
    # print()
    # print('='*15)
    args = sys.argv
    # print(args, len(args))
    # print('  '.join(list(sys.argv)))
    label = ['fake', 'real']
    datasets = ['ff++', 'cdf', 'dfdc', 'CelebDF-v2', 'dfdc_preview_set']
    if 'real' in args[2]:    idx = 1
    elif 'fake' in args[2]:  idx = 0
    else: print('Wrong Dir')
    print(args[2][-5:-1], '-------------------------------', idx, label[idx])
        
    dataset = ''
    if args[6] in datasets:
        dataset = 'dfdc' if 'dfdc' in args[6] else 'cdf'
        print(f'{dataset} -------------------------------------------------------')
        path = f'/media/data1/FaceForensics_Dec2020/CSIRO/test/{args[6]}/{label[idx]}/'
        jiwon_path = f"/media/data2/jiwon/CSIRO/lgrad/{dataset}/{label[idx]}/"
    else :
        print(f"[FALSE] {dataset} {args}")
        exit
        path = f"/media/data2/binh/CSIRO/collected_data_sok_single/{label[idx]}/"
        jiwon_path = f"/media/data2/jiwon/CSIRO/lgrad/collected_data/{label[idx]}/"
    dir = sorted(os.listdir(path))
    os.makedirs(jiwon_path, mode=0o777, exist_ok=True)
    save_dir = os.listdir(jiwon_path)
    print(len(dir), dir[0], dir[-1], "/////////////////////////////")
    
    count = -1
    
    for d in dir:
        count = count+1
        if ".pkl" in d: continue
        if ".txt" in d: continue
        if ".pth" in d: continue
        
        args[2] =path+d+'/'
        # if args[2].endswith(".txt"): continue
        args[3] =jiwon_path+d+'/'
        print(f"[{count}/{len(dir)}] ============================")
        
        if d in save_dir: 

            in_d_dir = len(os.listdir(args[2]))
            out_d_dir = len(os.listdir(args[3]))
            print(f'[{count}/{len(dir)}] [!] --------- {d} are already in OUPPUT DIR --- IN:{in_d_dir}, OUT:{out_d_dir}')
            
            if in_d_dir - 1 > out_d_dir: 
                print(f'[BUT] ----- Still running on {d}')
            else : continue
        # if 'mouth' in os.listdir(args[2]):
        #     print(f"[FORDER IS IN THERE] ----- {label[idx]} ---------- {args[2]}")
        #     continue
        print('=' *15)
        # print("[ARGS ALL] ------- ", args)
        # print("[ARGS_1] --------- ", args[1])
        print(f"--------- ", args[2])
        # print("[ARGS_3] --------- ", args[3])
        # print("[ARGS_4] --------- ", args[4])
        # print("[ARGS_5] --------- ", args[5])
        # continue
        main(args)
    print('='*30)
