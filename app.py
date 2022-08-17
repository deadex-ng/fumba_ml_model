import streamlit as st
from fastai.vision.all import *
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


def is_cat(x): return x[0].isupper() 

learn = load_learner('model.pkl')

categories = ('Dog','Cat')

def classify_image(img):
    pred,idx,probs =learn.predict(img)
    listt = [pred,idx,probs]
    return listt

# Designing the interface
st.title("Cat Image Classification App")
# For newline
st.write('\n')

image = Image.open("Cat.jpg")
show = st.image(image,use_column_width=True)

st.sidebar.title("Upload Image")

#Choose your own image
uploaded_file = st.sidebar.file_uploader(" ", type=['png','jpg', 'jpeg'])

if uploaded_file is not None:
    
    u_img = Image.open(uploaded_file)
    show.image(u_img, 'Uploaded Image', use_column_width=True)
    # We preprocess the image to fit in algorithm.
    im = PILImage.create(uploaded_file)
    
    im.thumbnail((192,192))

# For newline
st.sidebar.write('\n')
    
if st.sidebar.button("Click Here to Classify"):
    
    if uploaded_file is None:
        
        st.sidebar.write("Please upload an Image to Classify")
    
    else:
        
        with st.spinner('Classifying ...'):
            results = classify_image(im)
            time.sleep(3)
            st.success('Done!')
        st.sidebar.header("Algorithm classifies: ")
        dd = dict(zip(categories,map(float,results[2])))
        for cat,prob in dd.items():
            st.sidebar.write(cat,"{:.0%}".format(prob))
    
    
