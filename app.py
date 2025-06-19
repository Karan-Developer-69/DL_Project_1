import numpy as np
from flask import Flask, request, render_template
from keras.preprocessing import image
import tensorflow as tf
import os

app = Flask(__name__)
model = tf.keras.models.load_model('model.h5')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(300, 300))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)

        label = "Human" if classes[0] > 0.5 else "Horse"
        value = float(classes[0])

        return render_template('index.html', prediction=label, value=value, image_path=filepath)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)