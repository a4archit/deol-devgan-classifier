import tensorflow as tf
import streamlit as st 

from numpy import expand_dims
from tensorflow import keras 
from keras import Model
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout

def load_model_architecture():
    model = Sequential(name = "ajay_devgan_vs_sunny_deol")

    model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.15))

    model.add(Dense(1, activation='sigmoid'))
    model.build(input_shape=(None, 256, 256, 3))
    model.load_weights('/home/archit-elitebook/workarea/whole working/deep learning/deol_vs_devgan_clf/model_sa1.weights.h5')

    return model
    

def preprocess_image(img, image_path = None, target_size=(256, 256)):
    if img == False:
        img = tf.keras.utils.load_img(image_path, target_size=target_size)  # Load image
    img_array = tf.keras.utils.img_to_array(img)  # Convert to array
    img_array = img_array / 255.0  # Normalize (if required)
    img_array = expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array



if __name__ == "__main__":
    
    model = load_model_architecture()

    # image_path = "/home/archit-elitebook/workarea/whole working/deep learning/deol_vs_devgan_clf/testing_data/sunny_007"

    # input_image = preprocess_image(image_path)

    # print(round(model.predict(input_image).max()))


    # *************************** Streamlit [GUI] ***************************** #

    st.title("Deol Devgan classifier")

    input_img = st.file_uploader("Upload an image: ")

    st.image(input_img)

    processed_img = preprocess_image(input_img)


    st.info(round(model.predict(processed_img).max()))




