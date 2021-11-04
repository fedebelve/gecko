from gecko import settings
import tensorflow as tf
from tensorflow import keras 
from keras.applications. inception_v3 import InceptionV3
import os
import random
from django.conf import settings
from rest_framework.exceptions import APIException
import cv2
import pickle as pk
from gecko.settings import BASE_DIR

def create_model():
    random.seed(42)
    model_name = 'model_ft.008-0.592.h5'
    #self.graph = tf.get_default_graph()

    base_model = InceptionV3(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(299, 299, 3),
        include_top=False)


    inputs = tf.keras.Input(shape=(299, 299, 3))
    x = tf.keras.applications.inception_v3.preprocess_input(inputs)
    x = base_model(x, training=False) 

    x = keras.layers.GlobalAveragePooling2D()(x) # similar a flattear
    x = keras.layers.Dropout(0.2)(x) 
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)

    learning_rate_last_layer = 1e-2
    learning_rate_fine_tunning = 1e-4
    epsilon = 0.1

    model.compile(optimizer=keras.optimizers.Adam(learning_rate_last_layer, epsilon=epsilon),
            loss=tf.keras.losses.BinaryCrossentropy())        # creo que no son necesaros ni el loss ni el optimizer

    model.load_weights(os.path.join(settings.MODEL_ROOT, model_name))

    return model


from keras import backend as K

def accuracy_m(y_true, y_pred):
    y_true = K.ones_like(y_true)
    acc = K.mean(K.equal(y_true, K.round(y_pred)))
    return acc

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (all_positives + K.epsilon())
    return recall

# Crear custom las otras fp, fn, tn
def true_positives_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    return true_positives

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1score_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def load_model(model_name):
    model=tf.keras.models.load_model(os.path.join(settings.MODEL_ROOT, model_name), custom_objects={'f1score_m': f1score_m, 'precision_m': precision_m, 'true_positives_m': true_positives_m, 'recall_m': recall_m, 'accuracy_m': accuracy_m})
    print(f'Modelo {model_name} cargado.')
    return model

def load_pca(dir, pca_name):
    file=dir+'/'+pca_name
    model=pk.load(open(os.path.join(settings.PCA_ROOT, file),'rb'))
    print(f'PCA {pca_name} cargado.')
    return model

def fill(cant_bytes):
    resto = cant_bytes%4
    fill = ''
    if resto == 1:
        fill = '==='
    if resto == 2:
        fill = '=='
    if resto == 3:
        fill = '='

    return fill

def get_img_from_path(path):
    try:
        img = cv2.imread(path)
    except Exception as e:
        raise APIException("Error al leer la imagen")
    
    if img is None:
        raise APIException("Error al leer la imagen")

    return img


def get_paths(step, img_name):
    if step == 'checked':
        return f"{BASE_DIR}/tmp/{step}_images/{img_name}"

    path_299 = f"{BASE_DIR}/tmp/{step}_images/299/{img_name}.jpeg"
    path_380 = f"{BASE_DIR}/tmp/{step}_images/380/{img_name}.jpeg"
    return path_299, path_380

def remove_img_from(step, img_name):
    path_299, path_380 = get_paths(step, img_name)   
    os.remove(path_299)
    if not step == 'checked':
        os.remove(path_380)
    pass

def save_images(step, img_name, img_299, img_380):
    path_299, path_380 = get_paths(step, img_name)
    cv2.imwrite(path_299, img_299)
    cv2.imwrite(path_380, img_380)
    pass