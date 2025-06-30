from flask import Flask, request
from flask_cors import CORS
from keras.models import model_from_json
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

class_labels = {0: 'Benign', 1: 'Malignant'}

# load the pre-trained model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def load_image(path):
    image = cv2.imread(path)
    return image

@app.route("/classify", methods=["POST"])
def classify():
    image = request.files['image'].read()
    np_img = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (224, 224))
    img = img.reshape((1,224,224,3))

    prediction = loaded_model.predict(img)
    class_index = np.argmax(prediction)
    class_label = class_labels[class_index]
    probability = prediction[0][class_index] * 100
    
    return {'classification': class_label, 'probability': probability}

if __name__ == "__main__":
    app.run()
