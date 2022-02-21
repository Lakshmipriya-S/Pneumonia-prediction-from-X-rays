  
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import os
import cv2
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
from werkzeug.utils import secure_filename
import os

model_file = "model2v2.h5"
model = load_model(model_file)
app = Flask(__name__)
app.config["UPLOAD_FOLDER"]="static"


@app.route('/pneumonia1')
def pneumonia():
    return render_template('pneumonia1.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Pneumonia', methods=['POST'])
def upload_Pneumonia():
    uploaded_file = request.files['file']
    full_filename = secure_filename(uploaded_file.filename)
    uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], full_filename))
    path=os.path.join(app.config['UPLOAD_FOLDER'], full_filename)
    img = Image.open(path) # we open the image
    img_d = img.resize((224,224))
    # we resize the image for the model
    rgbimg=None
    #We check if image is RGB or not
    if len(np.array(img_d).shape)<3:
        rgbimg = Image.new("RGB", img_d.size)
        rgbimg.paste(img_d)
    else:
        rgbimg = img_d
    rgbimg = np.array(rgbimg,dtype=np.float64)
    rgbimg = rgbimg.reshape((1,224,224,3))
    predictions = model.predict(rgbimg)
    a = int(np.argmax(predictions))
    if a==1:
       a = "pneumonic"
    else:
       a="healthy"
    return render_template('context.html',filename = uploaded_file.filename, text=a)
if __name__ == '__main__':
   #app.run(debug=True)
    app.run(host="0.0.0.0",port=5000,threaded=False)    