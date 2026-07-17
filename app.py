import os
import pickle
from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# 1. Resolve path dynamically to avoid directory issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

model = None

# Helper function to generate a backup model if yours is missing
def create_fallback_model():
    print("⚠️ Creating a temporary dummy model for testing...")
    fallback = LogisticRegression()
    # Fit on minimal dummy data: 2 features, 2 classes
    fallback.fit([[0, 0], [1, 1]], [0, 1])
    try:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(fallback, f)
        print(f"✅ Created a temporary backup model at: {MODEL_PATH}")
    except Exception as e:
        print(f"Could not save backup model locally: {e}")
    return fallback

# 2. Try loading your real model, fall back to dummy if missing
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        print("🎉 Real 'model.pkl' loaded successfully!")
    else:
        print(f"⚠️ 'model.pkl' not found at {MODEL_PATH}.")
        model = create_fallback_model()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = create_fallback_model()


@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "message": "ML Model API is up and running!",
        "model_loaded": model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model is not initialized"}), 500
        
    try:
        data = request.get_json(force=True)
        # Expected input format: {"features": [val1, val2]}
        features = data.get('features')
        
        if not features:
            return jsonify({"error": "Missing 'features' key in request body"}), 400
            
        prediction = model.predict([features])
        
        return jsonify({
            "prediction": int(prediction[0]),
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Runs locally on http://127.0.0.1:5000
    app.run(debug=True)
