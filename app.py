from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
import re
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(image_path):
    try:
        img = Image.open(image_path).convert('RGB').resize((224, 224))
        img = np.array(img).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        return output
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def interpret_output(output):
    nail_conditions = [
        "Darier's disease", "Muehrcke's lines", "Alopecia areata", "Beau's lines", 
        "Bluish nail", "Clubbing", "Eczema", "Half and half nails (Lindsay's nails)",
        "Koilonychia", "Leukonychia", "Onycholysis", "Pale nail", "Red lunula", 
        "Splinter hemorrhage", "Terry's nail", "White nail", "Yellow nails", "Healthy nail"
    ]
    
    if output is None or not isinstance(output, np.ndarray) or output.ndim != 2:
        return "Unknown", 0.0
    
    index = int(np.argmax(output))
    confidence = float(np.max(output)) * 100
    return nail_conditions[index] if index < len(nail_conditions) else "Unknown", confidence

def get_recommendations(label):
    diet_map = {
        "healthy_nail": {"deficiency": "No Deficiency", "foods": ["Balanced Diet", "Hydration", "Fruits & Vegetables"]},
        "darier's_disease": {"deficiency": "Vitamin A, Zinc Deficiency", "foods": ["Carrots", "Spinach", "Eggs", "Dairy"]},
        "muehrcke's_lines": {"deficiency": "Protein Deficiency", "foods": ["Chicken", "Fish", "Eggs", "Dairy"]},
        "alopecia_areata": {"deficiency": "Iron, Vitamin D Deficiency", "foods": ["Red Meat", "Fortified Cereals", "Salmon"]},
        "beau_s_lines": {"deficiency": "Zinc Deficiency", "foods": ["Nuts", "Whole Grains", "Dairy"]},
        "bluish_nail": {"deficiency": "Oxygen Deficiency (Anemia)", "foods": ["Leafy Greens", "Red Meat", "Lentils"]},
        "eczema": {"deficiency": "Vitamin E, Omega-3 Deficiency", "foods": ["Avocados", "Nuts", "Fish"]},
        "koilonychia": {"deficiency": "Iron Deficiency (Anemia)", "foods": ["Spinach", "Lentils", "Quinoa"]},
        "leukonychia": {"deficiency": "Zinc Deficiency", "foods": ["Pumpkin Seeds", "Cashews", "Legumes"]},
        "pale_nail": {"deficiency": "Anemia", "foods": ["Iron-Rich Foods", "Beef", "Kale"]},
        "splinter_hemorrhage": {"deficiency": "Vitamin C Deficiency", "foods": ["Citrus Fruits", "Berries", "Bell Peppers"]},
        "white_nail": {"deficiency": "Protein Deficiency", "foods": ["Legumes", "Eggs", "Soy Products"]},
        "yellow_nails": {"deficiency": "Vitamin E Deficiency", "foods": ["Almonds", "Sunflower Seeds", "Olive Oil"]},
        "clubbing": {"deficiency": "Chronic Oxygen Deficiency (Possible Lung/Cardiac Issues)","foods": ["Leafy Greens", "Beets", "Iron-Rich Foods"]},
        "half_and_half_nails_lindsays_nails": {"deficiency": "Kidney Disease Indicator (Protein, Zinc Possible)","foods": ["Lean Protein", "Zinc-Rich Foods", "Low-Potassium Vegetables"]},
        "onycholysis": {"deficiency": "Iron, Biotin Deficiency (Thyroid Link)", "foods": ["Eggs", "Nuts", "Leafy Greens", "Whole Grains"]},
        "red_lunula": {"deficiency": "Possible Cardiovascular or Lupus-Related","foods": ["Heart-Healthy Foods", "Omega-3 Sources", "Antioxidants"]},
        "terry's_nail": {"deficiency": "Protein, Zinc Deficiency (Liver Disease)","foods": ["Lean Protein", "Zinc-Rich Foods", "Legumes"]}
    }
    
    # Convert label to dictionary key format
    key = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    print(f"Checking key: {key}")  # Debug line
    if key not in diet_map:
        print(f"Missing key: {key}")
    return diet_map.get(key, {"deficiency": "Consult a nutritionist", "foods": []})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if not file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        return jsonify({"error": "Invalid image format"})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    prediction = predict(filepath)
    if prediction is None:
        return jsonify({"error": "Prediction failed"})

    label, confidence = interpret_output(prediction)
    recommendations = get_recommendations(label)
    
    return jsonify({
        "deficiency": label,
        "confidence": round(confidence, 2),
        "diagnosis": recommendations["deficiency"],
        "foods": recommendations["foods"]
    })

if __name__ == '__main__':
    app.run(debug=True)
