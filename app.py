import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image, ImageOps

# === CONFIGURATION ===
UPLOAD_FOLDER = 'static/uploads'  # Changed to static/uploads for proper serving
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.getenv('MODEL_PATH', 'models/best_model.tflite')
TARGET_SIZE = (128, 128)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)  # Required for flashing messages

# Create required directories
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Global variable for the interpreter
interpreter = None

def load_model():
    global interpreter
    try:
        if os.path.exists(MODEL_PATH):
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            return True
        return False
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(path):
    try:
        # Load, convert to grayscale, crop, resize, normalize
        img = Image.open(path).convert('L')
        bw = img.point(lambda x: 0 if x<200 else 255, '1')
        if (bbox := bw.getbbox()):
            img = img.crop(bbox)
        img = img.resize(TARGET_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        # Match interpreter's expected shape
        return np.expand_dims(arr, axis=(0, -1))
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
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

                # Check if model is loaded
                if interpreter is None:
                    if not load_model():
                        return render_template('index.html', 
                                            error="Model not available. Please upload model file first.")

                # Prepare input & run inference
                input_data = preprocess_image(path)
                if input_data is None:
                    return render_template('index.html', 
                                        error="Error processing image. Please try another image.")

                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                preds = interpreter.get_tensor(output_details[0]['index'])
                
                # Extract top prediction
                idx = np.argmax(preds[0])
                conf = preds[0][idx]
                label = str(idx)  # replace with your class mapping if available
                
                return render_template('index.html',
                                    filename=filename,
                                    prediction=label,
                                    confidence=f"{conf:.2%}")
            except Exception as e:
                return render_template('index.html', 
                                    error=f"An error occurred: {str(e)}")

        flash('Invalid file type')
        return redirect(request.url)
    
    return render_template('index.html')

# To serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    load_model()  # Try to load model at startup
    app.run(host='0.0.0.0', port=port)
