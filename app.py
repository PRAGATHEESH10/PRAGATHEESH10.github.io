from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Setup upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
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
        "healthy_nail": {
            "deficiency": "No Deficiency",
            "foods": ["சமநிலை உணவு (Balanced Diet)", "தண்ணீர் (Hydration)", "மூலிகை காய்கறிகள் & பழங்கள் (Fruits & Vegetables) 🥦🍎"]
        },
        "darier's_disease": {
            "deficiency": "Vitamin A Deficiency, Zinc Deficiency",
            "foods": ["காரட் (Carrots 🥕)", "முருங்கை கீரை (Drumstick Leaves)", "முட்டை (Eggs 🥚)", "பாலாடை (Dairy 🥛)"]
        },
        "muehrcke's_lines": {
            "deficiency": "Protein Deficiency, Hypoalbuminemia",
            "foods": ["நாட்டு கோழி (Country Chicken 🍗)", "சேலா மீன் (Seer Fish 🐟)", "முட்டை (Eggs 🥚)", "பால் மற்றும் தயிர் (Milk & Curd)"]
        },
        "alopecia_areata": {
            "deficiency": "Iron Deficiency, Vitamin D Deficiency",
            "foods": ["மாடு இறைச்சி (Beef/Red Meat 🥩)", "போஷிக்கப்பட்ட தரிசி (Fortified Ragi/Thinai)", "வஞ்சரம் மீன் (Salmon or Seer Fish 🐟)"]
        },
        "beau_s_lines": {
            "deficiency": "Zinc Deficiency, Magnesium Deficiency",
            "foods": ["பருப்பு வகைகள் (Nuts 🥜)", "சோளம்/கம்பு (Millets/Whole Grains 🌾)", "பாலாடை (Dairy Products 🥛)"]
        },
        "bluish_nail": {
            "deficiency": "Oxygen Deficiency, Iron Deficiency (Anemia)",
            "foods": ["அமரந்த கீரை (Amaranth Leaves 🌿)", "மாடு இறைச்சி (Red Meat 🥩)", "பயறு வகைகள் (Lentils 🫘)"]
        },
        "eczema": {
            "deficiency": "Vitamin E Deficiency, Omega-3 Deficiency",
            "foods": ["வெண்ணை பருப்பு (Butter Beans/Avocados 🥑)", "முந்திரி/வேர்க்கடலை (Cashews & Groundnuts 🥜)", "சாளகிரி மீன் (Indian Mackerel 🐟)"]
        },
        "koilonychia": {
            "deficiency": "Iron Deficiency, Vitamin B12 Deficiency",
            "foods": ["முருங்கை கீரை (Drumstick Leaves 🌿)", "பயறு (Lentils 🫘)", "கோதுமை/சோளம் (Wheat/Quinoa 🌾)"]
        },
        "leukonychia": {
            "deficiency": "Zinc Deficiency, Calcium Deficiency",
            "foods": ["பூசணி விதைகள் (Pumpkin Seeds 🎃)", "முந்திரி (Cashews 🥜)", "பயறு வகைகள் (Legumes 🫘)"]
        },
        "pale_nail": {
            "deficiency": "Iron Deficiency, Folate Deficiency",
            "foods": ["இரும்புச் சத்து நிறைந்த உணவுகள் (Iron-Rich Foods 🥬)", "மாடு இறைச்சி (Beef 🥩)", "கீரை வகைகள் (Kale/Spinach 🌿)"]
        },
        "splinter_hemorrhage": {
            "deficiency": "Vitamin C Deficiency, Iron Deficiency",
            "foods": ["நார்த்தங்காய் (Citrus Fruits 🍊)", "நாவல் பழம் (Jamun/Berries 🍇)", "குடைமிளகாய் (Bell Peppers 🌶️)"]
        },
        "white_nail": {
            "deficiency": "Protein Deficiency, Liver Dysfunction",
            "foods": ["பயறு வகைகள் (Legumes 🫘)", "முட்டை (Eggs 🥚)", "சோயா பொருட்கள் (Soy Products)"]
        },
        "yellow_nails": {
            "deficiency": "Vitamin E Deficiency, Selenium Deficiency",
            "foods": ["பாதாம் (Almonds 🌰)", "சூரியகாந்தி விதைகள் (Sunflower Seeds 🌻)", "நல்லெண்ணெய் (Olive Oil 🫒)"]
        },
        "clubbing": {
            "deficiency": "Chronic Oxygen Deficiency, Iron Deficiency",
            "foods": ["அமரந்த கீரை (Amaranth Greens 🌿)", "சிவப்புக் கிழங்கு (Beets 🥬)", "இரும்பு சத்து நிறைந்த உணவுகள் (Iron-Rich Foods)"]
        },
        "half_and_half_nails_lindsays_nails": {
            "deficiency": "Kidney Dysfunction, Protein Deficiency, Zinc Deficiency",
            "foods": ["மீன்/முட்டை (Lean Protein 🐟🥚)", "வெந்தயம், முந்திரி (Zinc-Rich Foods 🌰)", "புடலங்காய், சுரைக்காய் (Low-Potassium Vegetables 🥒)"]
        },
        "onycholysis": {
            "deficiency": "Iron Deficiency, Biotin Deficiency, Thyroid Dysfunction",
            "foods": ["முட்டை (Eggs 🥚)", "வேர்க்கடலை (Groundnuts 🥜)", "அமரந்த கீரை (Greens 🌿)", "கம்பு/சோளம் (Whole Grains 🌾)"]
        },
        "red_lunula": {
            "deficiency": "Cardiovascular Disorders, Autoimmune Issues (Lupus)",
            "foods": ["இதயம் நலமான உணவுகள் (Heart-Healthy Foods ❤️)", "சாளகிரி மீன் (Omega-3 🐟)", "ஆமலா, நெல்லிக்காய் (Antioxidants 🍀)"]
        },
        "terry's_nail": {
            "deficiency": "Protein Deficiency, Zinc Deficiency, Liver Disease",
            "foods": ["மீன்/நாட்டு கோழி (Lean Protein 🐓)", "முந்திரி, வெந்தயம் (Zinc-Rich 🌰)", "பயறு வகைகள் (Legumes 🫘)"]
        }
    }

    key = label.lower().replace("'", "").replace(" ", "_").replace("(", "").replace(")", "")
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

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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
    app.run()
