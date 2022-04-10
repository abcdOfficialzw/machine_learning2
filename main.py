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
from keras.applications.vgg16 import VGG16
from streamlit_autorefresh import st_autorefresh
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
    # print the classification
    print('%s (%.2f%%)' % (label[1], label[2] * 100))
    #global detected
    #detected['frame%d' %count] = label[1]
    print('frame%d : ' %count, label[1])

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
      st.success("Saved File:{} to tempDir".format(uploaded_file.name))
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

  return
## Function to search for objects
def search_for_objects(search):
      found = False
      count = 0

      while count < len(objects) - 1:
            if objects[count] == search:
                  global search_results
                  search_results.append(objects[count].index())
                  found = True
      if found == False:
            st.error('The object you searched for is not in the video')
            return
      display_resulst(search_results)
## Function to display images found                  
def display_resulst(search_results):
      images = []
      count = 0
      while count < len(search_results):
            image = Image.open('./frames/frame%d' %count)
            images = images.append(image)
      st.image(images, 'Your result')
      

def main():
  

    """Object detection App"""

    st.title("Object Detection App")

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Object Detecting WebApp</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.title("Detect and classify ")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
    st.button('continue')
    st_autorefresh(1, 1, '0')

    temporary_location = False
    search_results = []

    if uploaded_file is not None:
        st_autorefresh(1, 1, '1')
        if os.path.exists('uploadedVideos'):     
          filename = 'uploadedVideos/' + str(save_uploadedfile(uploaded_file))
          ## Split video into frames
          st.info('upload successful, now splitting into frames')
          generate_frames(filename)
          st.info('video split successfully, bow detecting objects')
          ## Detect objects in frames
          global objects
          objects = []
          detect_Object()
          search_object = st.text_input('search')
          st.button('Search', onclick = search_for_objects(search_for_objects)):
        else:
              os.mkdir('uploadedVideos')
              return




if __name__ == '__main__':
    main()
