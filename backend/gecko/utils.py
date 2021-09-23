from gecko import settings
import tensorflow as tf
from tensorflow import keras 
from keras.applications. inception_v3 import InceptionV3
import os
import random
from django.conf import settings


def create_model():
    random.seed(42)
    model_name = 'model_ft.008-0.592.h5'
    #self.graph = tf.get_default_graph()
    print(settings.MODEL_ROOT)
    print(os.path.join(settings.MODEL_ROOT, model_name))

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
