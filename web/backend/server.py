from flask import Flask, request, jsonify, render_template
from routes import generate_image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('layout.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    text_description = data.get('description')
    image_url = generate_image(text_description)
    return jsonify({"image_url": image_url})

if __name__ == '__main__':
    app.run(debug=True)
