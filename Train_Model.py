import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.model_selection import train_test_split
import plotly.express as px
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Adamax
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

directory = r"E:\project\Project1\data_pattern"
labels = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def read_data(folder):
    label, paths = [], []
    for l in labels:
        path = f"{folder}/{l}/"
        folder_data = os.listdir(path)
        for image_path in folder_data:
            label.append(l)
            paths.append(os.path.join(folder, l, image_path))
    
    return label, paths

all_labels, all_paths = read_data(directory)

df = pd.DataFrame({
    'path': all_paths,
    'label': all_labels
})

train_df, dummy_df = train_test_split(df, train_size=0.8, random_state=123, shuffle=True, stratify=df['label'])
valid_df, test_df = train_test_split(dummy_df, train_size=0.5, random_state=123, shuffle=True, stratify=dummy_df['label'])

print("Train dataset : ", len(train_df), "Test dataset : ", len(test_df), "Validation dataset : ", len(valid_df))
print('Train dataset value count: \n', train_df['label'].value_counts())

labels = os.listdir(directory)
print("Labels:", labels)
labels.sort()
print("Sorted Labels:", labels)

px.histogram(train_df, x='label', barmode='group')

def apply_transform(image):
    angle = np.random.uniform(-40, 40)
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 0)
    alpha = 1.0 + np.random.uniform(-0.2, 0.2)
    beta = 0.0 + np.random.uniform(-0.2, 0.2)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    gamma = np.random.uniform(0.8, 1.2)
    image = np.clip((image / 255.0) ** gamma, 0, 1) * 255.0
    return image.astype(np.uint8)

def apply_augmentation(image):
    image = (image * 255.0).astype(np.uint8)
    augmented_image = apply_transform(image)
    return augmented_image.astype(np.float32) / 255.0

gen = ImageDataGenerator(
    preprocessing_function=apply_augmentation
)

train_gen = gen.flow_from_dataframe(
    train_df,
    x_col='path',
    y_col='label',
    target_size=(255, 255),
    seed=123,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
    batch_size=32
)

valid_gen = ImageDataGenerator().flow_from_dataframe(
    valid_df,
    x_col='path',
    y_col='label',
    target_size=(255, 255),
    seed=123,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False,
    batch_size=32
)

test_gen = ImageDataGenerator().flow_from_dataframe(
    test_df,
    x_col='path',
    y_col='label',
    target_size=(255, 255),
    seed=123,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False,
    batch_size=32
)

# CNN Model
model = Sequential([
    Convolution2D(32, kernel_size=3, padding='same', activation="relu", input_shape=(255, 255, 3)),
    MaxPooling2D(strides=2, pool_size=2, padding="valid"),
    Convolution2D(32, kernel_size=3, padding='same', activation="relu"),
    MaxPooling2D(strides=2, pool_size=2, padding="valid"),
    Dropout(0.5),
    Convolution2D(32, kernel_size=2, padding='same', activation="relu"),
    MaxPooling2D(strides=2, pool_size=2, padding="valid"),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.summary()

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="model_cnn.h5",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history_cnn = model.fit(
    x=train_gen,
    epochs=40,
    verbose=1,
    validation_data=valid_gen,
    validation_steps=None,
    shuffle=True,
    callbacks=[model_checkpoint_callback]
)

model.save("model_cnn_2.h5")
print("CNN model saved as model_cnn_2.h5")

# ResNet50 Model
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(255, 255, 3)
)
print('Created ResNet50 model')

for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers[173:]:
    layer.trainable = True

x1 = base_model.output
x2 = tf.keras.layers.GlobalAveragePooling2D()(x1)
x3 = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer="he_uniform")(x2)
x4 = tf.keras.layers.Dropout(0.4)(x3)
x5 = tf.keras.layers.Dense(512, activation='relu', kernel_initializer="he_uniform")(x4)
prediction = tf.keras.layers.Dense(10, activation='softmax')(x5)

final_model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction)

final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_checkpoint_callback_rs = tf.keras.callbacks.ModelCheckpoint(
    filepath="model_50.h5",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

history_resnet = final_model.fit(
    train_gen,
    epochs=20,
    validation_data=valid_gen,
    callbacks=[model_checkpoint_callback_rs]
)

final_model.save("model_50.h5")
print("ResNet50 model saved as model_50.h5")

# Plot CNN training results
acc = history_cnn.history['accuracy']
val_acc = history_cnn.history['val_accuracy']
loss = history_cnn.history['loss']
val_loss = history_cnn.history['val_loss']

epochs_range = range(len(history_cnn.history['accuracy']))

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('CNN Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('CNN Training and Validation Loss')
plt.show()

# Evaluate ResNet50 (use final_model)
def predictor(model, test_gen):
    classes = list(test_gen.class_indices.keys())
    class_count = len(classes)
    preds = model.predict(test_gen, verbose=1)
    errors = 0
    pred_indices = []
    test_count = len(preds)
    for i, p in enumerate(preds):
        pred_index = np.argmax(p)
        pred_indices.append(pred_index)
        true_index = test_gen.labels[i]
        if pred_index != true_index:
            errors += 1
    accuracy = (test_count - errors) * 100 / test_count
    ytrue = np.array(test_gen.labels, dtype='int')
    ypred = np.array(pred_indices, dtype='int')
    msg = f'There were {errors} errors in {test_count} tests for an accuracy of {accuracy:6.2f}'
    print(msg)
    cm = confusion_matrix(ytrue, ypred)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
    plt.xticks(np.arange(class_count) + 0.5, classes, rotation=90)
    plt.yticks(np.arange(class_count) + 0.5, classes, rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    clr = classification_report(ytrue, ypred, target_names=classes, digits=4)
    print("Classification Report:\n----------------------\n", clr)

def get_model_prediction(image_path):
    img = load_img(image_path, target_size=(255, 255))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    predictions = final_model.predict(x, verbose=0)  # Use final_model for ResNet50
    return labels[predictions.argmax()]

pred = []
for file in test_df['path'].values:
    pred.append(get_model_prediction(file))

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
random_index = np.random.randint(0, len(test_gen), 16)

for i, ax in enumerate(axes.ravel()):
    img_path = test_df['path'].iloc[random_index[i]]
    ax.imshow(load_img(img_path))
    ax.axis('off')
    if test_df['label'].iloc[random_index[i]] == pred[random_index[i]]:
        color = "green"
    else:
        color = "red"
    ax.set_title(f"True: {test_df['label'].iloc[random_index[i]]}\nPredicted: {pred[random_index[i]]}", color=color)

plt.tight_layout()
plt.show()
