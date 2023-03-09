#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle


# In[2]:


model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

# Add Layer Embedding
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#print(model.summary())



# In[3]:


def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)     #noramlised result

    return normalized_result

filenames = []
j=0
for file in os.listdir('/Users/manishsaini/Downloads/images'):
    if j<10000:
        filenames.append(os.path.join('/Users/manishsaini/Downloads/images',file))
        j+=1
    else:
        break
    


# In[4]:


filenames[0]


# In[5]:


## we already compiled model on traing data
feature_list = []
i=0
for file in tqdm(filenames):
    if i<10000:
        feature_list.append(extract_features(file,model))
        i+=1
    else:
        break
    


# In[6]:


pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))


# In[ ]:


from sklearn.neighbors import NearestNeighbors
import cv2
import pandas as pd

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

neighbors = NearestNeighbors(n_neighbors=6,algorithm='auto', metric='euclidean',n_jobs= -1)
neighbors.fit(feature_list)

normalized_result=extract_features("/Users/manishsaini/Downloads/slipper.jpg",model)

distances,indices = neighbors.kneighbors([normalized_result])

print(indices)
for file in indices[0][1:2]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output',cv2.resize(temp_img,(512,512)))
    cv2.waitKey(0)



# ## Code for web page ( User Interface)

# In[1]:


import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm


feature_list = np.array(pickle.load(open("/Users/manishsaini/embeddings.pkl",'rb')))
filenames = pickle.load(open('filenames.pkl','rb'))
                      
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('/Users/manishsaini/Downloads/images', uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='auto', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# steps
st.title('Fashion Recommender System')
# file upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # feature extract
        st.title(" Similar iamges are developed below")
        features = feature_extraction(os.path.join("/Users/manishsaini/Downloads/images",uploaded_file.name),model)
        #st.text(features)
        # recommendention
        indices = recommend(features,feature_list)
        # show
        col1,col2,col3,col4,col5 = st.beta_columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occured in file upload")


# In[ ]:




