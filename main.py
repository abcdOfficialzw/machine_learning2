"""
  Machine Learning Assignment
  ===========================

  Group Members:  Nyanhanga Takudzwa Titus R204543C
                  Lloyd Masuko R204541G

  Deployed on Streamlit
"""
## Importing Dependancies
import streamlit as st
import shutil
import cv2
import os
from PIL import Image
import numpy as np
from mailbox import ExternalClashError
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
import io
from streamlit_autorefresh import st_autorefresh
from keras.applications.vgg16 import VGG16
# load the model
model = VGG16()
from keras.preprocessing.image import load_img
# load an image from file
from keras.preprocessing.image import img_to_array
# convert the image pixels to a numpy array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

## Function to detect object
def detect_Object():
  count = 0
  while count < len(os.listdir('./frames')):
    image = load_img('frames/frame%d.jpg' %count, target_size=(224, 224))
    image = img_to_array(image)
    # convert the image pixels to a numpy array
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    object = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(object)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    print('frame%d :' %count, label[1])

    global objects
    objects.append(label[1])
    count = count + 1
    print(type(objects))
  return objects
## Function to save the uploaded file
def save_uploadedfile(uploaded_file):
    with open(os.path.join("uploadedVideos", uploaded_file.name), "wb") as f:
      f.write(uploaded_file.getbuffer())
      global filename
      filename = uploaded_file.name
      st.success("Saved File: {} to uploadedVideos".format(uploaded_file.name))
      st_autorefresh(1,1,'5654')
      return filename
## Function to split video into frames
def generate_frames(video):
  vidcap = cv2.VideoCapture(video)
  success, image = vidcap.read()
  count = 0
  while success:
    if os.path.exists('./frames'):
      cv2.imwrite("frames/frame%d.jpg" % count, image)  # save frame as JPEG file
      success, image = vidcap.read()
      print('Read a new frame: ', success)
      count += 1
    else:
      os.mkdir('frames')
      return
  st.success('Your video has successfully been split into frames, now detecting objects')
  st_autorefresh(1,1,'367')
  return
## Function to search for objects
def search_for_objects(search):
      st.info('Searching')
      found = False
      count = 0
      while count < len(objects) - 1:
            if objects[count] == search:
                  global search_results
                  search_results.append(objects[count].index())
                  found = True
                  count += 1

      if found == False:
            st.error('The object you searched for is not in the video')
            return
## Function to display images found                  
def display_resulst(search_results):
      images = []
      count = 0
      while count < len(search_results):
            image = Image.open('./frames/frame%d' %count)
            images = images.append(image)
      st.image(images, 'Your result')
      

def main():
    #"""Object detection App"""
    #st.title("Object Detection App")
    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:grey ;padding:10px">
    <h2 style="color:white;text-align:center;">Object Detecting WebApp</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.title("Detect and classify ")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
    if st.button('continue'):
      st_autorefresh(1, 1, '0')
      return

    temporary_location = False
    search_results = []

    if uploaded_file is not None:
        #st_autorefresh(1, 1, '1')
        if os.path.exists('uploadedVideos'):     
          filename = 'uploadedVideos/' + str(save_uploadedfile(uploaded_file))
          ## Split video into frames
          st_autorefresh(1,1,'3')
          generate_frames(filename)
          ## Detect objects in frames
          global objects
          objects = []
          detect_Object()
          st_autorefresh(1,1,'395')
          ## Search object
          search_object = st.text_input('search')
          if st.button('Search'):
            if search_for_objects(search_object):
                  display_resulst(search_results)
                  return
        else:
              ## create the directory
              os.mkdir('uploadedVideos')
              return
        return
    return  




if __name__ == '__main__':
    main()
