import uuid
from flask import Flask, jsonify, request

app = Flask(__name__)

keys = {}

@app.route('/generate_key', methods=['POST'])
def generate_key():
    username = request.json.get('username')
    if not username:
        return jsonify({'error': 'Username is required'}), 400

    api_key = str(uuid.uuid4())
    keys[username] = api_key
    return jsonify({'api_key': api_key})

@app.route('/validate_key', methods=['POST'])
def validate_key():
    username = request.json.get('username')
    api_key = request.json.get('api_key')
    if keys.get(username) == api_key:
        return jsonify({'valid': True})
    else:
        return jsonify({'valid': False})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
