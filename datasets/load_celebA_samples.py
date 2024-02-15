# script to load the celebA dataset

import os
import numpy as np
import pandas as pd
#import image reading library
from matplotlib import pyplot as plt

def load_celebA_samples():
    # get project directory
    prvs_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(os.getcwd(), 'celebA')
    # load the celebA_chosen_attributes_train.csv file
    attr_file = os.path.join(data_dir, 'celebA_chosen_attributes_train.csv')
    attr_df = pd.read_csv(attr_file, sep=',', header=0)
    # concat the columns of the dataframe
    attributes = attr_df.columns
    # get the image names
    train_img_names = []
    for attribute in attributes:
        train_img_names += list(attr_df[attribute].values)
    train_img_names = np.array(train_img_names).astype(str)
    # create the train_labels
    train_labels = np.zeros(train_img_names.shape[0])
    for i, attribute in enumerate(attributes):
        # train_labels[np.where(attr_df[attribute].isin(train_img_names))] = i
        train_labels[np.where(attr_df[attribute].isin(train_img_names))[0] + i * 5000] = i
    train_labels = train_labels.astype(int)
    # add the directory to the image names
    directory = np.zeros(train_img_names.shape[0]).astype(object)
    directory.fill(os.path.join(data_dir, 'img_align_celeba', ''))
    directory = directory.astype(str)
    train_img_names = np.char.add(directory, train_img_names)
    # load the celebA_chosen_attributes_test.csv file
    attr_file = os.path.join(data_dir, 'celebA_chosen_attributes_test.csv')
    attr_df = pd.read_csv(attr_file, sep=',', header=0)
    # concat the columns of the dataframe
    attributes = attr_df.columns
    # get the image names
    test_img_names = []
    for attribute in attributes:
        test_img_names += list(attr_df[attribute].values)
    test_img_names = np.array(test_img_names).astype(str)
    # create the test_labels
    test_labels = np.zeros(test_img_names.shape[0])
    for i, attribute in enumerate(attributes):
        # test_labels[np.where(attr_df[attribute].isin(test_img_names) )] = i
        test_labels[np.where(attr_df[attribute].isin(test_img_names))[0] + i * 1000] = i
    test_labels = test_labels.astype(int)
    # add the directory to the image names
    directory = np.zeros(test_img_names.shape[0]).astype(object)
    directory.fill(os.path.join(data_dir, 'img_align_celeba', ''))
    directory = directory.astype(str)
    test_img_names = np.char.add(directory, test_img_names)
    os.chdir(prvs_dir)
    return train_img_names, train_labels, test_img_names, test_labels

if __name__ == '__main__':
    train_img_names, train_labels, test_img_names, test_labels = load_celebA_samples()
    print(train_img_names.shape)
    print(train_labels.shape)
    print(test_img_names.shape)
    print(test_labels.shape)
    plt.figure()
    plt.title('First 10 images')
    # get data directory
    data_dir = os.path.join(os.getcwd(), 'celebA')
    # plot one image from each class in the training set in single figure
    imgs = train_img_names[::5000]
    labels = train_labels[::5000]
    for i, data in enumerate(zip(imgs, labels)):
        img, label = data
        img = plt.imread(img)
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.title(f'Class: {label}')
        plt.axis('off')
    plt.show()