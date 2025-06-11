import os
import base64
import json
import smtplib
from flask import Flask, request, jsonify
from flask_cors import CORS
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import cv2
import numpy as np
import tensorflow.lite as tflite

# Config
MODEL_PATH = 'animal_classifier_quant.tflite'
UPLOAD_FOLDER = 'images'
LOG_FILENAME = 'images_log.json'

# Email setup (use env variables on Render)
EMAIL_SENDER = os.environ.get('patilchetan64178@gmail.com', 'your_email@gmail.com')
EMAIL_PASSWORD = os.environ.get('erys vxjr vivr dbzs', 'your_gmail_app_password')
EMAIL_RECEIVER = os.environ.get('patilchetan65178@gmail.com', 'receiver_email@gmail.com')

# Setup
app = Flask(__name__)
CORS(app)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
log_path = os.path.join(UPLOAD_FOLDER, LOG_FILENAME)

# Load model
INTERPRETER = tflite.Interpreter(model_path=MODEL_PATH)
INTERPRETER.allocate_tensors()
input_details = INTERPRETER.get_input_details()
output_details = INTERPRETER.get_output_details()

ANIMAL_CLASSES = [
    "Armadillo", "Bear", "Bird", "Cow", "Crocodile", "Deer", "Elephant",
    "Goat", "Horse", "Jaguar", "Monkey", "Rabbit", "Skunk", "Tiger", "Wild Boar"
]

def predict_class(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (96, 96))
        input_tensor = np.expand_dims(img, axis=(0, -1)).astype(np.uint8)
        INTERPRETER.set_tensor(input_details[0]['index'], input_tensor)
        INTERPRETER.invoke()
        output = INTERPRETER.get_tensor(output_details[0]['index'])[0]
        return ANIMAL_CLASSES[int(np.argmax(output))]
    except Exception as e:
        print("Prediction error:", e)
        return "Unknown"

def send_email_notification(image_name, distance, prediction):
    body = f"""🚨 Animal Detected!

Image: {image_name}
Distance: {distance} cm
Predicted Class: {prediction}
"""
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = "🚨 Animal Detection Alert"
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print("✅ Email sent.")
    except Exception as e:
        print("❌ Email failed:", e)

def load_log():
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_log(log_data):
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)

@app.route('/upload_interrupt', methods=['POST'])
def upload_interrupt():
    try:
        data = request.get_json(force=True)
        image_b64 = data['image']
        distance = data['distance']
        image_name = data['imagename']

        image_path = os.path.join(UPLOAD_FOLDER, image_name)
        with open(image_path, 'wb') as f:
            f.write(base64.b64decode(image_b64))

        prediction = predict_class(image_path)

        log = load_log()
        log.append({
            "imagename": image_name,
            "distance_cm": distance,
            "classified_as": prediction
        })
        save_log(log)

        send_email_notification(image_name, distance, prediction)

        return jsonify({
            "status": "success",
            "classified_as": prediction,
            "distance": distance,
            "imagename": image_name
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)
