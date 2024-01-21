# %%
##############################################################
# This notebook is for training a CNN for Accident detection #
# Framework is tensorflow                                    #
# This is MY usual solution that works in Most situations    #
# with reasonable Accuracy                                   #
# Thanks                                                     #
##############################################################

# %%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from time import perf_counter
import os
  

# %%
import mlflow
from mlflow import pyfunc
import mlflow.keras

experiment_name = 'Accidents-Classification'
mlflow.set_experiment(experiment_name)

mlflow.keras.autolog()

# %%
## Defining batch specfications
batch_size = 100
img_height = 250
img_width = 250

# %%
## loading training set
training_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'data/train',
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size

)

# %%
## loading validation dataset
validation_ds =  tf.keras.preprocessing.image_dataset_from_directory(
    'data/val',
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size)


# %%
testing_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'data/test',
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size)

class_names = training_ds.class_names

## Configuring dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
training_ds = training_ds.cache().prefetch(buffer_size=AUTOTUNE)
testing_ds = testing_ds.cache().prefetch(buffer_size=AUTOTUNE)

# %%
# Defining Cnn

MyCnn = tf.keras.models.Sequential([
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(len(class_names), activation= 'softmax')
    ])

MyCnn.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %%
# retVal = MyCnn.fit(training_ds, validation_data= validation_ds, epochs = 1)

# mlflow.keras.autolog()

with mlflow.start_run(run_name="Cnn_model") as run:
    ## lets train our CNN
    # MyCnn.fit(training_ds, validation_data=validation_ds, epochs=10)
    MyCnn.fit(training_ds, validation_data=validation_ds, epochs=3)

# %%
## lets vizualize results on testing data
AccuracyVector = []

# load the model saved by mlflow tracking
pyfunc_model = pyfunc.load_model(run.info.artifact_uri + "/model")

plt.figure(figsize=(30, 30))
for images, labels in testing_ds.take(1):
    predictions = pyfunc_model.predict(images)
    predlabel = []
    prdlbl = []
    
    for mem in predictions:
        predlabel.append(class_names[np.argmax(mem)])
        prdlbl.append(np.argmax(mem))
    
    AccuracyVector = np.array(prdlbl) == labels
    for i in range(40):
        ax = plt.subplot(10, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title('Pred: '+ predlabel[i]+' actl:'+class_names[labels[i]] )
        plt.axis('off')
        plt.grid(True)

print("done running!!")
