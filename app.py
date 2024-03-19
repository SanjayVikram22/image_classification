from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
import gdown

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Configure uploads directory
UPLOAD_FOLDER = 'images_up'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Check if the model file exists and download it if not
model_file = "./my_model.keras"
if not os.path.isfile(model_file):
    print("Model needs to be downloaded")
    url = "https://drive.google.com/uc?id=1k-vW5jqL9UdNEe6mvltPm3B6IaQuPUMK"
    gdown.download(url, model_file, quiet=False)
else:
    print("File exists")

# Load your model here (make sure to provide the correct path)
model = load_model(model_file)

@app.route('/')
def upload_form():
    # Render the upload form
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_images():
    if request.method == 'POST':
        images = request.files.getlist('image')
        user_input = request.form['user_input']
        results = []

        for image in images:
            filename = secure_filename(image.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)

            # Process and predict each image
            processed_image = preprocess_image(filepath)
            prediction = model.predict(processed_image)
            decoded_prediction = decode_predictions(prediction, top=1)[0][0]  # Get top prediction
            label, confidence = decoded_prediction[1], decoded_prediction[2]

            if user_input.lower() in label.lower():
                results.append((filename, label, confidence))

        # Optionally clear the folder after processing
        # clear_folder_content(app.config['UPLOAD_FOLDER'])

        # Pass the results to the template
        return render_template('index.html', results=results)

def preprocess_image(image_path):
    # Load and preprocess the image
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    return image

def clear_folder_content(folder_path):
    # Clear the specified folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

if __name__ == '__main__':
    app.run(debug=True)
