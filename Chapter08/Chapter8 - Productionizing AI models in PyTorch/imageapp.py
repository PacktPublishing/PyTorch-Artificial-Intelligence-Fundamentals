from flask import Flask, request, jsonify
from image_classifier import create_model, predict_image

app = Flask(__name__)
model = create_model()

@app.route('/predict', methods=['POST'])
def predicted():
    if 'image' not in request.files:
        return jsonify({'error': 'Image not found'}), 400

    image = request.files['image'].read()
    object_name = predict_image(model, image)

    return jsonify({'object_name' : object_name})

if __name__ == '__main__':
	app.run(debug=True)
