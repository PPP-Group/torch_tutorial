import os

original_dataset_dir = './dataset'
classes_list = os.listdir(original_dataset_dir)

base_dir = './splitted'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'val')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)


for classes in classes_list:
    os.mkdir(os.path.join(train_dir, classes))
    os.mkdir(os.path.join(validation_dir, classes))
    os.mkdir(os.path.join(test_dir, classes))