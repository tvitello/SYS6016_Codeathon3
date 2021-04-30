### Code inspired by and based on https://github.com/mrdbourke/cs329s-ml-deployment-tutorial/
### Travis Vitello
### tjv9qh
import os
import json
import requests
import SessionState
import streamlit as st
import tensorflow as tf
import numpy as np
from codeathon_utils import load_and_prep_image, classes_and_models, update_logger, predict_json

tf.enable_eager_execution()

# Setup environment credentials (you'll need to change these)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "tjv9qh-codeathon-3-381435a8c0d0.json" # change for your GCP key
PROJECT = "tjv9qh-codeathon-3" # change for your GCP project
REGION = "us-central1" #"us-east4" # change for your GCP region (where your model is hosted)

### Streamlit code (works as a straigtht-forward script) ###
st.title("SYS6016: Codeathon 3")
st.header("Identify Whether a Person is Wearing a Face Mask or Not")
st.subheader("Travis Vitello \n tjv9qh")
st.write("ML Model based on manually-reconstructed AlexNet model with an SGD optimizer.  Training was performed with early stopping,\
         achieving an accuracy of 99.1% on masks vs. no masks.  The model was trained against Ashish Jangra's \
             'Face Mask 12K Image Dataset' available on Kaggle at https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset. \
                 This model consisted of a test set comprised of 483 mask vs. 509 no mask images, and a training\
                     set comprised of 5000 mask vs. 5000 no mask images.  Augmentation was previously applied.")
from PIL import Image
image = Image.open('alexnet.JPG')
st.image(image, caption='Fig. 1 --- Manually-Built AlexNet Model Used in this Study')

st.write("References:")
st.write("[1] “Welcome to Streamlit — Streamlit 0.81.0 Documentation.” Streamlit, docs.streamlit.io/en/stable. Accessed 30 Apr. 2021.")
st.write("[2] Jangra, Ashish. “Face Mask ~12K Images Dataset.” Kaggle, 26 May 2020, www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset.")
st.write("[3] Bourke, David. “Mrdbourke/Cs329s-Ml-Deployment-Tutorial.” GitHub, 2021, github.com/mrdbourke/cs329s-ml-deployment-tutorial.")
st.write("[4] Alake,  Richmond.  “Implementing  AlexNet  CNN  Architecture  Using  TensorFlow  2.0  and  Keras.”  Medium,14 Aug. 2020, towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98")
st.write("[5] Verdhan, Vaibhav. Computer Vision Using Deep Learning: Neural Network Architectures with Python and Keras.1st ed., Apress, 2021.")

@st.cache # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(image, model, class_names):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.
    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    
    print(" ")
    print(" ")
    print(" ")
    print(image)
    #### !!!!!!!!!!!!!!!!!!!!!!!!
    image = load_and_prep_image(image)
    # Turn tensors into int16 (saves a lot of space, ML Engine has a limit of 1.5MB per request)
    #image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    image = tf.expand_dims(image, axis=0)
    #image = np.expand_dims(image,axis=0)
    preds = predict_json(project=PROJECT,
                         region=REGION,
                         model=model,
                         instances=image)

    pred_class = class_names[tf.argmax(preds[0])]
    pred_conf = tf.reduce_max(preds[0])
    return image, pred_class, pred_conf

# Pick the model version
choose_model = st.sidebar.selectbox(
    "Pick model you'd like to use",
    ("Model 1 (2 mask classes)", )
)

# Model choice logic
if choose_model == "Model 1 (2 mask classes)":
    CLASSES = classes_and_models["model_1"]["classes"]
    MODEL = classes_and_models["model_1"]["model_name"]


# Display info about model and classes
if st.checkbox("Show classes"):
    st.write(f"You chose {MODEL}, these are the classes of images it can identify:\n", CLASSES)

# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Upload an image of person",
                                 type=["jpeg", "jpg", "png"])

# Setup session state to remember state of app so refresh isn't always needed
# See: https://discuss.streamlit.io/t/the-button-inside-a-button-seems-to-reset-the-whole-app-why/1051/11 
session_state = SessionState.get(pred_button=False)




# Create logic for app flow
if not uploaded_file:
    st.warning("Upload an image.")
    st.stop()
else:
    session_state.uploaded_image = uploaded_file.read()
    st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Predict")

# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True 

# And if they did...
if session_state.pred_button:
    session_state.image, session_state.pred_class, session_state.pred_conf = make_prediction(session_state.uploaded_image, model=MODEL, class_names=CLASSES)
    st.write(f"Prediction: {session_state.pred_class}, \
               Confidence: {session_state.pred_conf:.3f}")

# TODO: code could be cleaned up to work with a main() function...
# if __name__ == "__main__":
#     main()