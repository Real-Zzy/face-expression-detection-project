import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, AveragePooling2D,
    Flatten, Dense, Dropout
)

def build_cnn_model():
    variance_scaling = tf.keras.initializers.VarianceScaling(
        scale=1.0,
        mode='fan_avg',
        distribution='uniform',
        seed=None
    )
    
    zeros_initializer = tf.keras.initializers.Zeros()

    model = Sequential(name="Sequential")

    model.add(Conv2D(
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation='relu',
        padding='valid',
        data_format='channels_last',
        use_bias=True,
        kernel_initializer=variance_scaling,
        bias_initializer=zeros_initializer,
        input_shape=(48, 48, 1), 
        name='conv2d_1'
    ))

    model.add(MaxPooling2D(
        pool_size=(5, 5),
        strides=(2, 2),
        padding='valid',
        data_format='channels_last',
        name='max_pooling2d_1'
    ))

    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        padding='valid',
        kernel_initializer=variance_scaling,
        bias_initializer=zeros_initializer,
        name='conv2d_2'
    ))

    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        padding='valid',
        kernel_initializer=variance_scaling,
        bias_initializer=zeros_initializer,
        name='conv2d_3'
    ))

    model.add(AveragePooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='valid',
        name='average_pooling2d_1'
    ))

    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu',
        padding='valid',
        kernel_initializer=variance_scaling,
        bias_initializer=zeros_initializer,
        name='conv2d_4'
    ))

    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu',
        padding='valid',
        kernel_initializer=variance_scaling,
        bias_initializer=zeros_initializer,
        name='conv2d_5'
    ))

    model.add(AveragePooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='valid',
        name='average_pooling2d_2'
    ))

    model.add(Flatten(name='flatten_1'))

    model.add(Dense(
        units=1024,
        activation='relu',
        kernel_initializer=variance_scaling,
        bias_initializer=zeros_initializer,
        name='dense_1'
    ))

    model.add(Dropout(rate=0.2, name='dropout_1'))

    model.add(Dense(
        units=1024,
        activation='relu',
        kernel_initializer=variance_scaling,
        bias_initializer=zeros_initializer,
        name='dense_2'
    ))

    model.add(Dropout(rate=0.2, name='dropout_2'))

    model.add(Dense(
        units=7,
        activation='softmax',
        kernel_initializer=variance_scaling,
        bias_initializer=zeros_initializer,
        name='dense_3'
    ))

    return model
