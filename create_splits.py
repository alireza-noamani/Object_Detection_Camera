import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger

from sklearn.model_selection import train_test_split
import shutil
def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    # TODO: Implement function
    # Generating paths
    train_path = os.path.join(data_dir, 'train/')
    val_path = os.path.join(data_dir, 'val/')
    test_path = os.path.join(data_dir, 'test/')

    # Making new directories for train, val, and test
    os.mkdir(train_path)
    os.mkdir(val_path)
    os.mkdir(test_path)

    # Generating list of filenames in current directory
    files_path = os.path.join(data_dir,'*.tfrecord')
    filenames = [os.path.basename(x) for x in glob.glob(files_path)]

    # Split files into train (60%), val (20%), test (20%)
    train_val_files, test_files  = train_test_split(filenames, test_size=0.20)
    train_files, val_files = train_test_split(train_val_files, test_size=0.25)
    
    # Moving the files into new directories based on the split
    for file in train_files:
        current_dir = os.path.join(data_dir, file)
        new_dir = os.path.join(train_path, file)
        # os.rename(current_dir, new_dir)
        shutil.copy(current_dir, new_dir)

    for file in val_files:
        current_dir = os.path.join(data_dir, file)
        new_dir = os.path.join(val_path, file)
        # os.rename(current_dir, new_dir)
        shutil.copy(current_dir, new_dir)

    for file in test_files:
        current_dir = os.path.join(data_dir, file)
        new_dir = os.path.join(test_path, file)
        # os.rename(current_dir, new_dir)
        shutil.copy(current_dir, new_dir)



if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)