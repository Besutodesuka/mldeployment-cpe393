from flask import Flask, request, jsonify, abort
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "ML Model is Running"
#  requirement
# - The "features" key exists
# - Each input has exactly 4 float values
# - Invalid input returns a clear error message with HTTP 400
    
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        input_features = np.array(data["features"]).reshape(len(data["features"]), -1)
    except Exception as e:
        if "features" not in data.keys():
            abort(404, "features not found")
        try: 
            test = len(data["features"][0]) != 4
            if test:
                abort(404, "invalid feature size it should be list of 4 float values for example [[5.1, 3.5, 1.4, 0.2],[6.2, 3.4, 5.4, 2.3]]")
            else:
                abort(404,e)
        except:
            abort(404, "invalid feature size it should be list of 4 float values for example [[5.1, 3.5, 1.4, 0.2],[6.2, 3.4, 5.4, 2.3]]")
        
    prediction = model.predict(input_features)
    return jsonify({"prediction": prediction.tolist()})


@app.route("/predict_prob", methods=["POST"])
def predict_prob():
    data = request.get_json()
    input_features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict_proba(input_features)
    cls = np.argmax(prediction)
    return jsonify({"prediction":int(cls),
                    "confidence":float(prediction[0,cls])
                    })

# @app.route("/predict_prob", methods=["POST"])
# def predict_prob():
#     data = request.get_json()
#     input_features = np.array(data["features"]).reshape(len(data["features"]), -1)
#     prediction = model.predict_proba(input_features)
#     cls = np.argmax(prediction, axis = -1)
#     return jsonify({"prediction":cls.tolist(),
#                     "confidence":prediction[:,cls].tolist()
#                     })

@app.route('/health', methods=['GET'])
def health():
    return jsonify(status="ok")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000) 


