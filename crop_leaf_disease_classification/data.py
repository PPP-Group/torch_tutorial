import os
import shutil
import math
from data_dir import classes_list, original_dataset_dir, train_dir, validation_dir, test_dir

for classes in classes_list:
    path = os.path.join(original_dataset_dir, classes)
    fnames = os.listdir(path)

    train_size = math.floor(len(fnames)*0.6)
    validation_size = math.floor(len(fnames)*0.2)
    test_size = math.floor(len(fnames)*0.2)

    train_fnames = fnames[:train_size]
    print(f"Train Size ({classes}): {len(train_fnames)}")
    for fname in train_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(os.path.join(train_dir, classes), fname)
        shutil.copyfile(src, dst)
    
    validation_fnames = fnames[train_size:(validation_size+train_size)]
    print(f'Validation Size ({classes}): {len(validation_fnames)}')
    for fname in validation_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(os.path.join(validation_dir, classes), fname)
        shutil.copyfile(src, dst)
    
    test_fnames = fnames[(train_size+validation_size):(validation_size+train_size+test_size)]
    print(f"Test Size({classes}): {len(test_fnames)}")
    for fname in test_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(os.path.join(test_dir, classes), fname)
        shutil.copyfile(src, dst)
