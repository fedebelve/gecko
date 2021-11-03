########################################################################
#
# Functions for preprocessing data-set.
#
# Implemented in Python 3.5
#
########################################################################

from gecko.settings import BASE_DIR
import tensorflow as tf
import os
import sys
import cv2
import matplotlib
matplotlib.use('agg')
from pylab import array, arange, uint8
#from PIL import ImageStat
import PIL
import math
import numpy as np
from gecko.settings import BASE_DIR, RN_VALIDATOR_MODEL, RN_INCEPTION_MODEL
from rest_framework.exceptions import APIException
from gecko.utils import get_img_from_path
########################################################################


def _increase_contrast(image):
    """
    Helper function for increasing contrast of image.
    """
    # Create a local copy of the image.
    copy = image.copy()

    maxIntensity = 255.0
    x = arange(maxIntensity)

    # Parameters for manipulating image data.
    phi = 1.3
    theta = 1.5
    y = (maxIntensity/phi)*(x/(maxIntensity/theta))**0.5

    # Decrease intensity such that dark pixels become much darker,
    # and bright pixels become slightly dark.
    copy = (maxIntensity/phi)*(copy/(maxIntensity/theta))**2
    copy = array(copy, dtype=uint8)

    return copy


def _find_contours(image):
    """
    Helper function for finding contours of image.

    Returns coordinates of contours.
    """
    # Increase constrast in image to increase changes of finding
    # contours.
    processed = _increase_contrast(image)

    # Get the gray-scale of the image.
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # Detect contour(s) in the image.
    cnts = cv2.findContours(
        gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # At least ensure that some contours were found.
    if len(cnts) > 0:
        # Find the largest contour in the mask.
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Assume the radius is of a certain size.
        if radius > 100:
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            return (center, radius)


def _get_filename(file_path):
    """
    Helper function to get filename including extension.
    """
    return file_path.split("/")[-1]


def _resize_and_center_fundus(image, diameter):
    """
    Helper function for scale normalizing image.
    """
    copy = image.copy()

    # Find largest contour in image.
    contours = _find_contours(image)

    # Return unless we have gotten some result contours.
    if contours is None:
        return None

    center, radius = contours

    # Calculate the min and max-boundaries for cropping the image.
    x_min = max(0, int(center[0] - radius))
    y_min = max(0, int(center[1] - radius))
    z = int(radius*2)
    x_max = x_min + z
    y_max = y_min + z

    # Crop the image.
    copy = copy[y_min:y_max, x_min:x_max]

    # Scale the image.
    fx = fy = (diameter / 2) / radius
    copy = cv2.resize(copy, (0, 0), fx=fx, fy=fy)

    # Add padding to image.
    shape = copy.shape

    # Get the border shape size.
    top = bottom = int((diameter - shape[0])/2)
    left = right = int((diameter - shape[1])/2)

    # Add 1 pixel if necessary.
    if shape[0] + top + bottom == diameter - 1:
        top += 1

    if shape[1] + left + right == diameter - 1:
        left += 1

    # Define border of the image.
    border = [top, bottom, left, right]

    # Add border.
    copy = cv2.copyMakeBorder(copy, *border,
                              borderType=cv2.BORDER_CONSTANT,
                              value=[0, 0, 0])
    # Return the image.
    return copy


def _get_image_paths(images_path):
    """
    Helper function for getting file paths to images.
    """
    return [os.path.join(images_path, fn) for fn in os.listdir(images_path)]


def _resize_and_center_fundus_all(image_paths, save_path, diameter, verbosity):
    # Get the total amount of images.
    num_images = len(image_paths)
    success = 0

    # For each image in the specified directory.
    for i, image_path in enumerate(image_paths):
        if verbosity > 0:
            # Status-message.
            msg = "\r- Preprocessing image: {0:>6} / {1}".format(
                    i+1, num_images)

            # Print the status message.
            sys.stdout.write(msg)
            sys.stdout.flush()

        try:
            # Load the image and clone it for output.
            image = cv2.imread(os.path.abspath(image_path), -1)

            processed = _resize_and_center_fundus(image, diameter=diameter)

            if processed is None:
                print("Could not preprocess {}...".format(image_path))
            else:
                # Get the save path for the processed image.
                image_filename = _get_filename(image_path)
                image_jpeg_filename = "{0}.jpg".format(os.path.splitext(
                                        os.path.basename(image_filename))[0])
                output_path = os.path.join(save_path, image_jpeg_filename)

                # Save the image.
                cv2.imwrite(output_path, processed,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                success += 1
        except AttributeError as e:
            print(e)
            print("Could not preprocess {}...".format(image_path))

    return success


########################################################################


def resize_and_center_fundus(save_path=None, images_path=None, image_paths=None,
                    image_path=None, diameter=299, verbosity=1):
    """
    Function for resizing and centering fundus in image.

    :param save_path:
        Required. Saves preprocessed image to the given path.

    :param images_path:
        Optional. Path to directory in where images reside.

    :param image_path:
        Optional. Single path to image.

    :param image_paths:
        Optional. List of paths to images.

    :param diameter:
        Optional. Result diameter of fundus. Defaults to 299.

    :return:
        Nothing.
    """
    if save_path is None:
        raise ValueError("Save path not specified!")

    save_path = os.path.abspath(save_path)

    if image_paths is not None:
        return _resize_and_center_fundus_all(image_paths=image_paths,
                                             save_path=save_path,
                                             diameter=diameter,
                                             verbosity=verbosity)

    elif images_path is not None:
        # Get the paths to all images.
        image_paths = _get_image_paths(images_path)
        # Scale all images.
        return _resize_and_center_fundus_all(image_paths=image_paths,
                                             save_path=save_path,
                                             diameter=diameter,
                                             verbosity=verbosity)

    elif image_path is not None:
        return _resize_and_center_fundus_all(image_paths=[image_path],
                                             save_path=save_path,
                                             diameter=diameter,
                                             verbosity=verbosity)


def resize(images_paths, size=299):
    """
    Function for resizing images.

    :param images_paths:
        Required. Paths to images.

    :param size:
        Optional. Size to which resize to. Defaults to 299.

    :return:
        Nothing.
    """
    for image_path in images_paths:
        image = cv2.imread(image_path)

        # Resize the image.
        image = cv2.resize(image, (size, size))

        # Save the image.
        cv2.imwrite(image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def rescale_min_1_to_1(image):
    """
    Rescale image to [-1, 1].

    :param image:
        Required. Image tensor.

    :return:
        Scaled image.
    """
    # Image must be casted to float32 first.
    image = tf.cast(image, tf.float32)
    # Rescale image from [0, 255] to [0, 2].
    image = tf.multiply(image, 1. / 127.5)
    # Rescale to [-1, 1].
    return tf.subtract(image, 1.0)


def rescale_0_to_1(image):
    """
    Rescale image to [0, 1].

    :param image:
        Required. Image tensor.

    :return:
        Scaled image.
    """
    return tf.image.convert_image_dtype(image, tf.float32)


def brightness_level(img_path):

    img = PIL.Image.open(img_path)
    stat = PIL.ImageStat.Stat(img)
    r, g, b = stat.mean

    return math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))  


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def ben_transformation(images_path):
  dirs = os.listdir(images_path)
  for item in dirs:
    if os.path.isfile(images_path+item):
      f, e = os.path.splitext(images_path+item)
      img = load_ben_color(images_path+item)
      image = Image.fromarray(img)
      image.save(f + '.jpg', quality=95) # es la maxima calidad de PIL
  print(f"Transformacion Ben hecha al directorio {images_path}")

def load_ben_color(path, sigmaX=10):
    #pre_process_img = cv2.imread(path)
    pre_process_img=get_img_from_path(path)
    image = cv2.cvtColor(pre_process_img, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.addWeighted( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
    
    return image

def pre_process_image(image_path, diameter = 299):

    success = 0
    try:

        #image = cv2.imread(os.path.abspath(image_path), -1)
        image = get_img_from_path(image_path)
        pre_processed_image = _resize_and_center_fundus(image, diameter=diameter)

        if pre_processed_image is None:
            print("Could not preprocess {}...".format(image))
        else:
            success += 1
            return pre_processed_image

    except Exception as e:
        print(e)
        print("Could not preprocess {}...".format(image))
        raise APIException("Imagen no encontrada")

    return success

def process_image(path):
    #process_img = cv2.imread(path)
    image = get_img_from_path(path)
    img = cv2.resize(image, (299,299), 3)
    imgg = img.reshape(1, 299, 299, 3)
    result = RN_INCEPTION_MODEL.predict(imgg)

    return result[0][0]

def is_retinography_img(img_path):
    pre_processed_image = pre_process_image(img_path, 224)
    img = pre_processed_image.reshape(1, 224, 224, 3)
    
    if RN_VALIDATOR_MODEL.predict(img) < 0.5:
        return True

def check_brightness_level(img_path):
    return (25 < brightness_level(img_path) < 150)


def validate(img_path):
    is_retinography = is_retinography_img(img_path)
    brightness_level_ok = check_brightness_level(img_path)

    return brightness_level_ok, is_retinography