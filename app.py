import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image, ImageOps
import keras
from tensorflow.keras import backend as K


UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
TARGET_SIZE = (128, 128)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24) 

# Create required directories
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('models', exist_ok=True)

model = keras.models.load_model("./models/ocr_model.keras", compile=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))

    return np.array(label_num)


def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret += alphabets[ch]
    return ret


def decode_prediction(pred):
    # Get the most probable character index at each time step
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    decoded, _ = K.ctc_decode(pred, input_length=input_len, greedy=True) # type: ignore
    decoded = K.get_value(decoded[0])[0] # type: ignore

    # Convert index sequence to string
    result = num_to_label(decoded)
    return result


def preprocess_image(img_path):
    try:
        # Open and convert image to grayscale
        with Image.open(img_path).convert("L") as img:
            img = np.array(img)

        h, w = img.shape
        final_img = np.ones((64, 256), dtype=np.uint8) * 255

        # Crop if needed
        img = img[: min(h, 64), : min(w, 256)]
        final_img[: img.shape[0], : img.shape[1]] = img

        
        final_img = np.transpose(final_img)[:, ::-1]

        
        final_img = final_img.astype(np.float32) / 255.0

        
        final_img = final_img.reshape(1, 256, 64, 1)

        return final_img

    except Exception as e:
        print(f"[ERROR] Failed to preprocess image: {e}")
        return None


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)
                img = preprocess_image(path)

                pred = model.predict(img)
                decoded_text = decode_prediction(pred)

                return render_template(
                    "index.html", prediction=decoded_text, filename=file.filename
                )
            except Exception as e:
                return render_template('index.html', 
                                    error=f"An error occurred: {str(e)}")

        flash('Invalid file type')
        return redirect(request.url)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3232))
    app.run(host='0.0.0.0', port=port)
