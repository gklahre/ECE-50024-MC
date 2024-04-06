import tensorflow as tf
import numpy as np
import PIL
import matplotlib.pyplot as plt
import pathlib
import csv

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def get_labels():
    labels = []
    
    g = open("category.csv")
    label_dict = dict()
    inverse_dict = dict()
    lines = g.readlines() 
    for line in lines:
        line = line.split(',')
        label_dict[line[1]]= line[0]
        inverse_dict[line[0]] = line[1]
    g.close()
    f = open("mytrain.csv")
    lines = f.readlines()

    for line in lines:
        parts = line.split(',')
        try: 
            int_lab = int(label_dict[parts[2]])
        except:
            print("start")
            int_lab = 0
        labels.append(int_lab)

    return labels
        
my_data = pathlib.Path("mytrain").with_suffix('')

batch_size = 32
img_height = 200
img_width = 200
#my_labels = get_labels()

training_data = tf.keras.utils.image_dataset_from_directory(my_data,validation_split=0.2,subset="training",seed=200,image_size=(img_height, img_width), batch_size=batch_size)
validation_data = tf.keras.utils.image_dataset_from_directory(my_data,validation_split=0.2,subset="validation",seed=200,image_size=(img_height, img_width), batch_size=batch_size)

#print(training_data.class_names)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = training_data.cache().shuffle(500).prefetch(buffer_size = AUTOTUNE)
val_ds = validation_data.cache().prefetch(buffer_size = AUTOTUNE)
#normalization_layer = layers.Rescaling(1./255)
"""
data_aug = Sequential([
    
    tf.keras.layers.RandomFlip("horizontal", input_shape=(200, 200, 3)),
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])
model = Sequential([
    data_aug,
    tf.keras.layers.Conv2D(16,3,padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32,3,padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64,3,padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.Flatten(),
    layers.Dense(2000, activation='relu'),
    layers.Dense(1600, activation='relu'),
    layers.Dense(1200, activation='relu'),
    layers.Dense(800, activation='relu'),
    layers.Dense(400, activation='relu'),
    layers.Dense(100)
])
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.summary()
"""
model = tf.keras.models.load_model('model_2_1.keras')
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='ch.m2.tf',monitor='val_accuracy',mode='max',save_best_only=True)

epochs = 5
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,callbacks=[model_checkpoint_callback])

model.save('model_2_2.keras')


#TODO: Save model\
# Convert the model.
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#tflite_model = converter.convert()


# Save the model.
#with open('model_simple.tflite', 'wb') as f:
  
  #f.write(tflite_model)
