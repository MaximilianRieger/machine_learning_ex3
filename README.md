# Setup:
- 1) extract datasets to the datasets folder
    - CelbA dataset should be in the `datasets/celebA` folder with the images in the `img_align_celeba` folder
    - CIFAR-10 dataset should be in the `datasets/cifar-10-batches-py` folder
- 3) run split_celebA.py to create the datasplit for the celebA dataset
- 4) create the histograms for the datasets by running the `create_histograms.py` script

# Training classic models:
- 1) run the classification scripts in the first steps folder

# Training the deep models:
- 1) run the main script in the second step folder with the arguments for the model and dataset
- 2) run the evaluate script in the second step folder with the arguments for the trained model path and dataset