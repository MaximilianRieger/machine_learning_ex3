# this script samples the celebA dataset for specific attributes and saves the samples as images and a list of new attributes as a csv file

import os
import numpy as np
import pandas as pd
import tqdm

#copy file from source to destination
def copyFile(source, dest):
    with open(source, 'rb') as f:
        with open(dest, 'wb') as f2:
            f2.write(f.read())


chosen_attributes = ['Smiling', 'Blond_Hair', 'Black_Hair', 'Brown_Hair', 'Gray_Hair', 'Pale_Skin', 'Narrow_Eyes', 'Rosy_Cheeks', 'Eyeglasses', 'Wearing_Lipstick']

if __name__ == '__main__':
    # get project directory
    prvs_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(os.getcwd(), 'celebA')

    # load the list_attr_celeba.csv file
    attr_file = os.path.join(data_dir, 'list_attr_celeba.csv')
    attr_df = pd.read_csv(attr_file, sep=',', header=0)

    # get the attributes
    attributes = attr_df.columns[1:]

    attributes_image_dict = {}

    # sample the dataset for specific attributes
    for attribute in chosen_attributes:
        # get the rows of the dataframe that have the specific attributes
        rows = attr_df[(attr_df[attribute] == 1)]

        # sample 100 images from the rows
        sample = rows.sample(5000, random_state=42)
        # get the image names
        img_names = sample['image_id'].values

        # !!! Do not copy the images, just save the names of the images in the csv file

        # save the images
        # save_dir = os.path.join(data_dir, "train", attribute)
        # os.makedirs(save_dir, exist_ok=True)
        # make progress bar
        # progress = tqdm.tqdm(img_names, desc=f'{attribute}', unit='image')
        # for img_name in progress:
        #     # copy the images to the new directory
        #     copyFile(os.path.join(data_dir, "img_align_celeba", img_name), os.path.join(save_dir, img_name))

        attributes_image_dict.update({attribute: img_names})

    # save the dictionary as a csv file
    df = pd.DataFrame(attributes_image_dict)
    df.to_csv(os.path.join(data_dir,'celebA_chosen_attributes_train.csv'), index=False)
    print('Train data saved')

    # now for the test set
    # sample the dataset for specific attributes
    for attribute in chosen_attributes:
        # get the rows of the dataframe that have the specific attributes
        rows = attr_df[(attr_df[attribute] == 1)]

        # sample 100 images from the rows
        sample = rows.sample(1000, random_state=42)
        # get the image names
        img_names = sample['image_id'].values
        # !!! Do not copy the images, just save the names of the images in the csv file

        # save the images
        # save_dir = os.path.join(data_dir, "test", attribute)
        # os.makedirs(save_dir, exist_ok=True)
        # progress = tqdm.tqdm(img_names, desc=f'{attribute}', unit='image')
        # for img_name in progress:
        #     # copy the images to the new directory
        #     copyFile(os.path.join(data_dir, "img_align_celeba", img_name), os.path.join(save_dir, img_name))


        attributes_image_dict.update({attribute: img_names})

    # save the dictionary as a csv file
    df = pd.DataFrame(attributes_image_dict)
    df.to_csv(os.path.join(data_dir,'celebA_chosen_attributes_test.csv'), index=False)
    print('Test data saved')