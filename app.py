from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

app = Flask(__name__)
UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Load TFLite model with absolute path
MODEL_PATH = Path(__file__).parent / "model.tflite"
interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get model input shape dynamically
input_shape = tuple(input_details[0]['shape'][1:3])  # (height, width)

def predict(image_path):
    try:
        img = Image.open(image_path).convert('RGB').resize(input_shape)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        return output
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None

def interpret_output(output):
    nail_conditions = [
        "Darier's disease", "Muehrcke's lines", "Alopecia areata", "Beau's lines", 
        "Bluish nail", "Clubbing", "Eczema", "Half and half nails (Lindsay's nails)",
        "Koilonychia", "Leukonychia", "Onycholysis", "Pale nail", "Red lunula", 
        "Splinter hemorrhage", "Terry's nail", "White nail", "Yellow nails", "Healthy nail"
    ]
    
    if output is None or not isinstance(output, np.ndarray):
        return "Unknown", 0.0
    
    index = np.argmax(output)
    confidence = float(np.max(output)) * 100
    return nail_conditions[index] if index < len(nail_conditions) else "Unknown", confidence

def get_recommendations(label):
    diet_map = {
        "healthy_nail": {"deficiency": "No Deficiency", "foods": ["Balanced Diet", "Hydration", "Fruits & Vegetables"]},
        "dariers_disease": {"deficiency": "Vitamin A, Zinc Deficiency", "foods": ["Carrots", "Spinach", "Eggs", "Dairy"]},
        # ... (keep other entries but fix keys to match formatting)
        "terrys_nail": {"deficiency": "Protein, Zinc Deficiency", "foods": ["Lean Protein", "Zinc-Rich Foods", "Legumes"]}
    }
    
    key = label.lower().replace("'", "").replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    return diet_map.get(key, {"deficiency": "Consult a nutritionist", "foods": []})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    if not file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        return jsonify({"error": "Invalid image format"}), 400

    try:
        filepath = UPLOAD_FOLDER / file.filename
        file.save(filepath)
        
        prediction = predict(filepath)
        if prediction is None:
            return jsonify({"error": "Prediction failed"}), 500
            
        label, confidence = interpret_output(prediction)
        recommendations = get_recommendations(label)
        
        return jsonify({
            "deficiency": label,
            "confidence": round(confidence, 2),
            "diagnosis": recommendations["deficiency"],
            "foods": recommendations["foods"]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
