import os

from datasets.load_celebA_samples import load_celebA_samples
import cv2 as cv
import numpy as np
import time
import tqdm

if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_celebA_samples()

    # check if sift features are already extracted
    if not os.path.exists(os.path.join(os.getcwd(), 'descriptors.npy')):
        # print('Extracting SIFT features...')
        # create progressbar
        pbar = tqdm.tqdm(train_data, unit='images', desc='Extracting SIFT features')
        descriptors_to_use = 50
        sift = cv.SIFT_create()
        # get shape for the array
        dscs = np.zeros(shape=(len(train_data), descriptors_to_use, 128))
        start = time.time()
        for i, img_path in enumerate(pbar):
            # load image
            img = cv.imread(img_path)
            #resize image to halve the size
            img = cv.resize(img, (0, 0), fx=0.8, fy=0.8)
            _, desc = sift.detectAndCompute(img, None)
            # cap the number of descriptors to use and pad with zeros if there are not enough
            if desc is None:
                print('No descriptors found for', img_path)
                continue
            elif len(desc) < descriptors_to_use:
                desc = np.concatenate((desc, [np.zeros(128)] * (descriptors_to_use - len(desc))), axis=0)
            elif len(desc) >= descriptors_to_use:
                desc = np.array(desc[:descriptors_to_use])
            dscs[i] = desc


        end = time.time()
        print('Done extracting SIFT features, time:', end - start, 'seconds')
        # descriptors = np.array(descriptors)
        np.save('descriptors.npy', dscs)
    else:
        print('Loading SIFT features...')
        descriptors = np.load('descriptors.npy')
        print('SIFT features loaded')

    print('done')


