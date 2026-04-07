from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2

app = Flask(__name__)
model = load_model("new_model.keras")

IMAGE_SIZE = (224, 224)

class_names = [
    "Pomelo_blight",
    "Pomelo__healthy",
    "Pomelo__YellowLeaf__Curl_Virus",
    "Pomelo__Target_Spot",
    "Pomelo_Spider_mites_Two_spotted_spider_mite",
    "Pomelo_Septoria_leaf_spot",
    "Pomelo_Leaf_Mold",
    "Pomelo_Late_blight",
    "Pomelo_Early_blight",
    "Pomelo_Bacterial_spot",
]

fertilizer_suggestions_en = {
    "Pomelo_blight": "Apply chlorothalonil-based fungicide and remove infected leaves.",
    "Pomelo__healthy": "No treatment needed. Maintain proper watering and nutrients.",
    "Pomelo__YellowLeaf__Curl_Virus": "Control whiteflies using neem oil and remove infected plants.",
    "Pomelo__Target_Spot": "Use fungicides like chlorothalonil and improve air circulation.",
    "Pomelo_Spider_mites_Two_spotted_spider_mite": "Use neem oil or insecticidal soap.",
    "Pomelo_Septoria_leaf_spot": "Apply copper fungicide and avoid overhead watering.",
    "Pomelo_Leaf_Mold": "Maintain low humidity and use fungicide.",
    "Pomelo_Late_blight": "Use fungicides and remove infected parts immediately.",
    "Pomelo_Early_blight": "Apply mancozeb or chlorothalonil.",
    "Pomelo_Bacterial_spot": "Use copper bactericide and remove infected leaves."
}

fertilizer_suggestions_ta = {
    "Pomelo_blight": "குளோரோத்தாலோனில் போன்ற பூஞ்சைநாசினி பயன்படுத்தவும் மற்றும் பாதிக்கப்பட்ட இலைகளை அகற்றவும்.",
    "Pomelo__healthy": "சிகிச்சை தேவையில்லை. சரியான நீர்ப்பாசனம் மற்றும் உரம் கொடுக்கவும்.",
    "Pomelo__YellowLeaf__Curl_Virus": "வெள்ளை ஈக்களை கட்டுப்படுத்த நீம் எண்ணெய் பயன்படுத்தவும் மற்றும் பாதிக்கப்பட்ட தாவரங்களை அகற்றவும்.",
    "Pomelo__Target_Spot": "பூஞ்சைநாசினி பயன்படுத்தவும் மற்றும் காற்றோட்டத்தை மேம்படுத்தவும்.",
    "Pomelo_Spider_mites_Two_spotted_spider_mite": "நீம் எண்ணெய் அல்லது பூச்சி நாசினி சோப்பு பயன்படுத்தவும்.",
    "Pomelo_Septoria_leaf_spot": "செம்பு அடிப்படையிலான பூஞ்சைநாசினி பயன்படுத்தவும் மற்றும் மேலிருந்து நீர் ஊற்றுவதை தவிர்க்கவும்.",
    "Pomelo_Leaf_Mold": "ஈரப்பதத்தை குறைத்து பூஞ்சைநாசினி பயன்படுத்தவும்.",
    "Pomelo_Late_blight": "பூஞ்சைநாசினி பயன்படுத்தி பாதிக்கப்பட்ட பகுதிகளை அகற்றவும்.",
    "Pomelo_Early_blight": "மாங்கோசெப் அல்லது குளோரோத்தாலோனில் பயன்படுத்தவும்.",
    "Pomelo_Bacterial_spot": "செம்பு அடிப்படையிலான மருந்து பயன்படுத்தவும் மற்றும் பாதிக்கப்பட்ட இலைகளை அகற்றவும்."
}


def preprocess_image(file):
    img_array = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    image = cv2.resize(image, IMAGE_SIZE)
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    image = preprocess_image(file)
    preds = model.predict(image)

    print("Prediction shape:", preds.shape)
    print("Raw predictions:", preds)

    idx = int(np.argmax(preds))
    confidence = round(float(np.max(preds)) * 100, 2)

    if idx >= len(class_names):
        predicted_class = "Healthy"
    else:
        predicted_class = class_names[idx]

    suggestion_en = fertilizer_suggestions_en.get(predicted_class, "No treatment needed. Maintain proper watering and nutrients.")
    suggestion_ta = fertilizer_suggestions_ta.get(predicted_class, "சிகிச்சை தேவையில்லை. சரியான நீர்ப்பாசனம் மற்றும் உரம் கொடுக்கவும்.")

    return jsonify({
        "class": predicted_class,
        "confidence": confidence,
        "suggestion_en": suggestion_en,   # ✅ key names match predict.html
        "suggestion_ta": suggestion_ta    # ✅ key names match predict.html
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)