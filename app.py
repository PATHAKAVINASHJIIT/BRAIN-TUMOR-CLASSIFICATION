from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
import os
# Create a Flask app instance
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for image upload
@app.route('/upload', methods=['POST'])
def upload_image():
    print("Hi")
    # Get the uploaded file from the request object
    file = request.files['image']
    if not file:
        return 'No file uploaded.'
    
    test_image = Image.open(file.stream)

    # test_image = tf.keras.utils.load_img('Brain-MRI-Classification/Brain-MRI/predict//'+
    #                                      os.listdir("Brain-MRI-Classification/Brain-MRI/predict")[i], 
    #                                                 target_size = (150, 150))
    print(test_image)
    test_image = test_image.resize((150,150))
    loaded_model = keras.models.load_model('my_model.h5')
    test_image = tf.keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    print("Hi ",test_image)
    result = loaded_model.predict(test_image)
    
    

    class_idx = np.argmax(result, axis=1)[0]  # get the index of the predicted class
#   print(class_idx)
    class_labels = ['GLIOMA',  'NO', 'PITUATARY']  # define your class labels
    class_label = class_labels[class_idx]
    print(result)
    
    print('Predicted class:', class_label)

   
    # Load the saved model
    
    
    # Render the prediction result using a template
    return render_template('result.html', tumor_type=class_label)

# Run the Flask app
if __name__ == '__main__':
    app.run()

