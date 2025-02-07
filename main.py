""" 
Deol Devgan Classifier
======================
    This project is build to classify the image of Sunny Deol and Ajay Devgan.
    This project is based on CNN neural networks.
    There are only around 15 Millions weights used in this model.
    For more information you should visit: https://www.github.com/a4archit
""" 



# importing dependencies
import streamlit as st 
import os 

from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras.utils import load_img, img_to_array
from numpy import expand_dims
from tensorflow import keras 
from keras import Sequential
from keras import Model




st.set_page_config(
        page_title="Deol Devgan Classifier",
        page_icon="🤖"
)

#  Sidebar  
st.sidebar.title("About the developer")
st.sidebar.divider()
st.sidebar.write("I am creating this web application with \
**Streamlit** and build an **Ajay Devgan & Sunny Deol Classifier Model**, \
this model based on the **CNN** (Convolutional Neural Networks)")
st.sidebar.write("You can check my social media accounts: ")
st.sidebar.write("[Website](https://a4archit.github.io/my-portfolio)")
st.sidebar.write("[Kaggle](https://www.kaggle.com/architty108)")
st.sidebar.write("[Github](https://www.github.com/a4archit)")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/archit-tyagi-191323296)")





# creating function that build architecture and load trained parameters
def load_model_architecture():

    # creating an instance of sequential class
    model = Sequential(name = "ajay_devgan_vs_sunny_deol")

    # first convolutional layer (input layer)
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    # second convolutional layer
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    # third convolutional layer
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    # fourth convolutional layer
    model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    # Flatten layer
    model.add(Flatten())

    # first fullyconnected layer
    model.add(Dense(126, activation='relu'))
    model.add(Dropout(0.15))

    # output layer
    model.add(Dense(1, activation='sigmoid'))


    # loading model weights
    # model.load_weights('/workspaces/deol-devgan-classifier/model_sa2.weights.h5')
    model.load_weights(os.path.join(os.getcwd(), "model_sa2.weights.h5"))


    return model
    




# this method will preprocess uploaded image and make sure image ready for prediction
def preprocess_image(image_path: str, target_size=(256, 256)) :
    # performing operation
    img = load_img(image_path, target_size=target_size)  # Load image
    img_array = img_to_array(img)  # Convert to array
    img_array = img_array / 255.0  # Normalizing
    img_array = expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array







if __name__ == "__main__":
    
    # spinner will rotate until model setup successfully
    with st.spinner("Loading model..."):
        # creating mdoel instance
        model = load_model_architecture()



    # *************************** Streamlit [GUI] ***************************** #

    st.title("Deol Devgan classifier") # setting title

    # taking image from user
    uploaded_file = st.file_uploader("Upload a image file (supported formats: png, jpg, jpeg) either of Sunny Deol or Ajay Devgan")

    # if user upload an image
    if uploaded_file is not None:


        # rotate spinner until image successfully setup
        with st.spinner("Setting up loading image..."):

            # showing uploaded image
            st.image(uploaded_file, caption="Uploaded image")

            # Save the file temporarily
            file_path = os.path.join("tempDir", uploaded_file.name)
            
            # Ensure directory exists
            os.makedirs("tempDir", exist_ok=True)

            # Write the file to the directory
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # demonstrating when file successfully setup
            st.success(f"File saved at: {file_path}")


        # rotate spinner until image has been completely preprocessed
        with st.spinner("Preprocessing Image..."):

            # make sure that image ready for prediction
            processed_img = preprocess_image(f"tempDir/{uploaded_file.name}")


        # rotate spinner until model make prediction and prediction is not ready for diaplay
        with st.spinner("Prediction..."):

            # predict uploaded image
            prediction = model.predict(processed_img)

            # extract information from prediction
            predicted_class = round(prediction.max())

            # decoding labels
            prediction_label = "ajay devgan" if predicted_class == 0 else "sunny deol"
            

        st.divider()

        # display prediction
        output_markdown = f"##### According to my knowledge he is {prediction_label.title()}"
        st.markdown(output_markdown)




