import cv2
import numpy as np
import pickle
import os
import seaborn as sns
from tensorflow import keras as kr
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from keras.models import Sequential, Model, load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from keras import backend as K



def get_CAM(processed_image, predicted_label):
    """
    This function is used to generate a heatmap for a sample image prediction.

    Args:
        processed_image: any sample image that has been pre-processed using the
                       `preprocess_input()`method of a keras model
        predicted_label: label that has been predicted by the network for this image

    Returns:
        heatmap: heatmap generated over the last convolution layer output
    """
    # we want the activations for the predicted label
    predicted_output = model.output[:, predicted_label]

    # choose the last conv layer in your model
    last_conv_layer = model.get_layer('conv_7b_ac')

    # get the gradients wrt to the last conv layer
    grads = K.gradients(predicted_output, last_conv_layer.output)[0]

    # take mean gradient per feature map
    grads = K.mean(grads, axis=(0,1,2))
    # Define a function that generates the values for the output and gradients
    evaluation_function = K.function([model.input], [grads, last_conv_layer.output[0]])

    # get the values
    grads_values, conv_ouput_values = evaluation_function([processed_image])

    # iterate over each feature map in yout conv output and multiply
    # the gradient values with the conv output values. This gives an
    # indication of "how important a feature is"
    for i in range(512): # we have 512 features in our last conv layer
        conv_ouput_values[:,:,i] *= grads_values[i]

    # create a heatmap
    heatmap = np.mean(conv_ouput_values, axis=-1)

    # remove negative values
    heatmap = np.maximum(heatmap, 0)

    # normalize
    heatmap /= heatmap.max()

    return heatmap



def show_random_sample(idx):
    """
    This function is used to select a random sample from the validation dataframe.
    It generates prediction for the same. It also stores the heatmap and the intermediate
    layers activation maps.

    Arguments:
        idx: random index to select a sample from validation data

    Returns:
        activations: activation values for intermediate layers
    """
    # select the sample and read the corresponding image and label
    sample_image = cv2.imread(valid_df.iloc[idx]['image'])
    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
    sample_image = cv2.resize(sample_image, (img_rows, img_cols))
    sample_label = valid_df.iloc[idx]["label"]

    # pre-process the image
    sample_image_processed = np.expand_dims(sample_image, axis=0)
    sample_image_processed = preprocess_input(sample_image_processed)

    # generate activation maps from the intermediate layers using the visualization model
    activations = vis_model.predict(sample_image_processed)

    # get the label predicted by our original model
    pred_label = np.argmax(model.predict(sample_image_processed), axis=-1)[0]

    # choose any random activation map from the activation maps
    sample_activation = activations[0][0,:,:,32]

    # normalize the sample activation map
    sample_activation-=sample_activation.mean()
    sample_activation/=sample_activation.std()

    # convert pixel values between 0-255
    sample_activation *=255
    sample_activation = np.clip(sample_activation, 0, 255).astype(np.uint8)

    # get the heatmap for class activation map(CAM)
    heatmap = get_CAM(sample_image_processed, pred_label)
    heatmap = cv2.resize(heatmap, (sample_image.shape[0], sample_image.shape[1]))
    heatmap = heatmap *255
    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    super_imposed_image = heatmap * 0.5 + sample_image
    super_imposed_image = np.clip(super_imposed_image, 0,255).astype(np.uint8)

    f,ax = plt.subplots(2,2, figsize=(15,8))
    ax[0,0].imshow(sample_image)
    ax[0,0].set_title(f"True label: {sample_label} \n Predicted label: {pred_label}")
    ax[0,0].axis('off')

    ax[0,1].imshow(sample_activation)
    ax[0,1].set_title("Random feature map")
    ax[0,1].axis('off')

    ax[1,0].imshow(heatmap)
    ax[1,0].set_title("Class Activation Map")
    ax[1,0].axis('off')

    ax[1,1].imshow(super_imposed_image)
    ax[1,1].set_title("Activation map superimposed")
    ax[1,1].axis('off')
    plt.show()

    return activations



def visualize_intermediate_activations(layer_names, activations):
    """
    This function is used to visualize all the itermediate activation maps

    Arguments:
        layer_names: list of names of all the intermediate layers we chose
        activations: all the intermediate activation maps
    """
    assert len(layer_names)==len(activations), "Make sure layers and activation values match"
    images_per_row=16

    for layer_name, layer_activation in zip(layer_names, activations):
        nb_features = layer_activation.shape[-1]
        size= layer_activation.shape[1]

        nb_cols = nb_features // images_per_row
        grid = np.zeros((size*nb_cols, size*images_per_row))

        for col in range(nb_cols):
            for row in range(images_per_row):
                feature_map = layer_activation[0,:,:,col*images_per_row + row]
                feature_map -= feature_map.mean()
                feature_map /= feature_map.std()
                feature_map *=255
                feature_map = np.clip(feature_map, 0, 255).astype(np.uint8)

                grid[col*size:(col+1)*size, row*size:(row+1)*size] = feature_map

        scale = 1./size
        plt.figure(figsize=(scale*grid.shape[1], scale*grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(grid, aspect='auto', cmap='viridis')
    plt.show()


img_height = 299
img_width = 299
batch_size = 32



train_ds = kr.utils.image_dataset_from_directory(
    "Covid19-dataset/train",
    seed = 123,
    image_size = (img_height,img_width),
    validation_split = 0.2,
    subset = "training"
)

test_ds = kr.utils.image_dataset_from_directory(
    "Covid19-dataset/test",
    seed = 123,
    image_size = (img_height,img_width),
    validation_split = 0.2,
    subset = "validation"
)

def count(counts, batch):
  features, labels = batch
  class_1 = labels == 1
  class_1 = tf.cast(class_1, tf.int32)

  class_0 = labels == 0
  class_0 = tf.cast(class_0, tf.int32)

  class_2 = labels == 2
  class_2 = tf.cast(class_0, tf.int32)
  counts['class_0'] += tf.reduce_sum(class_0)
  counts['class_1'] += tf.reduce_sum(class_1)
  counts['class_2'] += tf.reduce_sum(class_2)

  return counts

counts = train_ds.take(10).reduce(
    initial_state={'class_0': 0, 'class_1': 0, 'class_2' : 0},
    reduce_func = count)

counts = np.array([counts['class_0'].numpy(),
                   counts['class_1'].numpy(),
                   counts['class_2']]).astype(np.float32)

fractions = counts/counts.sum()
print(fractions)




def class_func(features, label):
  return label

resample_ds = (
    train_ds
    .unbatch()
    .rejection_resample(class_func, target_dist=[0.33,0.33,0.33],
                        initial_dist=fractions)
    .batch(32))



base_model = tf.keras.applications.InceptionResNetV2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=(img_width,img_height,3),
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

base_model.trainable = False
base_model.summary()
optimizer = kr.optimizers.Adam(learning_rate=0.0005)


inputs = kr.Input(shape=(img_width,img_height,3))
mean = np.array([127.5]*3)
var = mean ** 2

norm_layer = kr.layers.experimental.preprocessing.Normalization(mean=mean,variance=var)
x = norm_layer(inputs)

x = base_model(x)

x = kr.layers.Dense(256)(x)
x = kr.layers.Dense(64)(x)
output = kr.layers.Dense(3, activation="softmax")(x)
model_fin = kr.Model(inputs,output)
model_fin.summary()


model_fin.compile(
    optimizer="Adam",
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)
model_fin.summary()


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights = True)
# checkpoint to save model
chkpt = ModelCheckpoint(filepath="model1", save_best_only=True)

history1 = model_fin.fit(
    train_ds,
    validation_data=test_ds,
    epochs=8,
    callbacks=[early_stop],
    shuffle=True
)

print("done")





# get the training and validation accuracy from the history object
train_acc = history1.history['accuracy']
valid_acc = history1.history['val_accuracy']

# get the loss
train_loss = history1.history['loss']
valid_loss = history1.history['val_loss']

# get the number of entries
xvalues = np.arange(len(train_acc))

# visualize
f,ax = plt.subplots(1,2, figsize=(10,5))
ax[0].plot(xvalues, train_loss)
ax[0].plot(xvalues, valid_loss)
ax[0].set_title("Loss curve")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("loss")
ax[0].legend(['train', 'validation'])

ax[1].plot(xvalues, train_acc)
ax[1].plot(xvalues, valid_acc)
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("accuracy")
ax[1].legend(['train', 'validation'])

plt.show()

#VISUALISING

# select all the layers for which you want to visualize the outputs and store it in a list
outputs = [layer.output for layer in model_fin.layers[3:]]
print(outputs)

# Define a new model that generates the above output
vis_model = Model(model_fin.input, outputs)

# check if we have all the layers we require for visualization
vis_model.summary()

# store the layer names we are interested in
layer_names = []
for layer in outputs:
    layer_names.append(layer.name.split("/")[0])


print("Layers to be used for visualization: ")
print(layer_names)


activations= show_random_sample(123)
visualize_intermediate_activations(activations=activations, layer_names=layer_names)


