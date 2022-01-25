import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    val_segments = []
    with open('validation.txt') as f:
        for line in f:
            val_segments.append(line.strip())
    
    file_list = glob.glob(os.path.join(data_dir, 'training_and_validation', '*.tfrecord'))
      
    for file in file_list:
        start = file.find('segment')
        end = file.find('_with_camera_labels')
        
        if file[start:end] in val_segments:
            os.rename(file, os.path.join(data_dir, 'val', file[start:]))
        else:
            os.rename(file, os.path.join(data_dir, 'train', file[start:]))

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)
