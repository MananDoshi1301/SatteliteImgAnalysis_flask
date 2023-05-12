from flask import Flask, render_template, request, session
import io
import torch
import numpy as np
import pandas as pd
import os
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
from PIL import Image
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.metrics


import os
import sys
import numpy as np
import re
import cv2 as op
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from os import listdir
import pandas as pd
from keras.layers import Dense, Dropout, Input, add, Conv2D, BatchNormalization, MaxPooling2D, Conv2DTranspose, Activation, Concatenate
from tensorflow import keras
from time import time
from tqdm import tqdm
from keras import backend as K

# Defining upload folder path
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
# # Define allowed files
ALLOWED_EXTENSIONS = {'tiff', 'png', 'jpg', 'jpeg'}

# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name for template path
# The default folder name for static files should be "static" else need to mention custom folder for static path
app = Flask(__name__, template_folder='templateFiles',
            static_folder='staticFiles')
# Configure upload folder for Flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Center crop padded image / mask to original image dims
def crop_image(image, target_image_dims=[1500, 1500, 3]):

    target_size = target_image_dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    if padding < 0:
        return image

    return image[
        padding:image_size - padding,
        padding:image_size - padding,
        :,
    ]

# Perform colour coding on the reverse-one-hot outputs


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis=-1)
    return x


@app.route('/')
def home():
    return render_template('index_upload_and_display_image.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        #
        file = request.files['file']

        #
        img_bytes = file.read()

        #
        my_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                [0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])])

        #
        image_op = Image.open(io.BytesIO(img_bytes))
        # image_tensor = my_transforms(image_op).unsqueeze(0)
        image_tensor = my_transforms(image_op)

        # Prereq
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        select_classes = ['background', 'road']

        class_dict = pd.read_csv("./label_class_dict.csv")
        # Get class names
        class_names = class_dict['name'].tolist()
        # Get class RGB values
        class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()

        # Get RGB values of required classes
        select_class_indices = [class_names.index(
            cls.lower()) for cls in select_classes]
        select_class_rgb_values = np.array(class_rgb_values)[
            select_class_indices]

        # Preprocess in function
        # test_dataset_vis[idx][0] == ip image
        image_vis = crop_image(image_tensor.astype('uint8'))
        # image_vis = crop_image(np.array(image_tensor, dtype=np.uint8))

        # data = np.array(image_tensor, dtype=np.uint8)
        # x_tensor = torch.from_numpy(image_tensor).to(DEVICE).unsqueeze(0)

        x_tensor = torch.from_numpy(image_vis).to(DEVICE).unsqueeze(0)

        # Loading Model
        best_model = torch.load('./best_model.pth', map_location=DEVICE)

        # Predict test image
        pred_mask = best_model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()

        # Convert pred_mask from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask, (1, 2, 0))

        # Get prediction channel corresponding to road
        pred_road_heatmap = pred_mask[:, :, select_classes.index('road')]
        pred_mask = crop_image(colour_code_segmentation(
            reverse_one_hot(pred_mask), select_class_rgb_values))
        # # # # Main image generated

        return render_template('show_image.html', user_image=img_file_path)
