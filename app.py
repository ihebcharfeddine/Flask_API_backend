from flask import Flask, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load your TensorFlow model
model = tf.keras.models.load_model('model/model.h5')

@app.route('/classify', methods=['GET'])
def classify():
    try:
        # Define test features for local testing
        test_features = [
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
        ]
        
        # Ensure test features match the model's expected input shape
        if len(test_features) != model.input_shape[1]:
            print("Error: Invalid test features length")
            return jsonify({"error": "Invalid test features length"}), 400

        # Convert test features to numpy array and reshape for the model
        features_array = np.array(test_features).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(features_array)
        
        # Get the predicted class
        predicted_class = int(np.argmax(prediction, axis=1)[0]) 
        prediction_probs = prediction[0].tolist()

        # Print results to the terminal
        print("Predicted Class:", predicted_class)
        print("Prediction Probabilities:", prediction_probs)

        return jsonify({
            "predicted_class": predicted_class,
            "prediction_probabilities": prediction_probs
        })
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
