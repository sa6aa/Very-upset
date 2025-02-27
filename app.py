import uuid
import os
import logging
from datetime import timedelta
from functools import wraps
from typing import Dict, Any

from flask import Flask, jsonify, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity
)
from flasgger import Swagger
from pydantic import BaseModel
from dotenv import load_dotenv
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from cachetools import TTLCache

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'sqlite:///database.db')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'super-secret')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt'}

# Initialize extensions
db = SQLAlchemy(app)
jwt = JWTManager(app)
swagger = Swagger(app)
limiter = Limiter(app=app, key_func=get_remote_address)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Initialize cache
cache = TTLCache(maxsize=100, ttl=300)

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    api_key = db.Column(db.String(120), unique=True, nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

class SavedText(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    file_path = db.Column(db.String(200), unique=True, nullable=False)

# Create tables
with app.app_context():
    db.create_all()

# Load DeepSeek models
try:
    # Text generation model
    deepseek_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")
    deepseek_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")
    
    # Code generation pipeline
    code_generator = pipeline(
        "text-generation",
        model=deepseek_model,
        tokenizer=deepseek_tokenizer,
        device=0 if os.getenv('USE_GPU') == 'true' else -1
    )
    
    # Sentiment analysis model
    sentiment_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-bert-base-sent")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("deepseek-ai/deepseek-bert-base-sent")
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model=sentiment_model,
        tokenizer=sentiment_tokenizer
    )
except Exception as e:
    logging.error(f"Error loading DeepSeek models: {str(e)}")
    raise e

# Initialize other pipelines
translator = pipeline('translation', model='Helsinki-NLP/opus-mt-mul-en')
summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
ner_pipeline = pipeline('ner', model='dslim/bert-base-NER')
grammar_checker = pipeline('text2text-generation', model='vennify/t5-base-grammar-correction')

# Pydantic models
class TextRequest(BaseModel):
    text: str
    max_length: int = 100

class TranslationRequest(BaseModel):
    text: str
    target_lang: str = 'en'

class CodeRequest(BaseModel):
    prompt: str
    max_length: int = 200

class FileUploadRequest(BaseModel):
    description: str

# Utility functions
def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Decorators
def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        current_user = User.query.filter_by(username=get_jwt_identity()).first()
        if not current_user or not current_user.is_admin:
            return jsonify({'error': 'Admin privileges required'}), 403
        return fn(*args, **kwargs)
    return wrapper

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Auth routes
@app.route('/register', methods=['POST'])
def register():
    """
    Register new user
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: UserRegistration
          required:
            - username
          properties:
            username:
              type: string
              example: user123
    responses:
      201:
        description: User created successfully
    """
    username = request.json.get('username')
    if not username:
        return jsonify({'error': 'Username is required'}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already exists'}), 400

    api_key = str(uuid.uuid4())
    new_user = User(username=username, api_key=api_key)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({
        'message': 'User created successfully',
        'api_key': api_key
    }), 201

@app.route('/login', methods=['POST'])
def login():
    """
    User login
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: UserLogin
          required:
            - username
            - api_key
          properties:
            username:
              type: string
            api_key:
              type: string
    responses:
      200:
        description: Login successful
    """
    username = request.json.get('username')
    api_key = request.json.get('api_key')

    user = User.query.filter_by(username=username, api_key=api_key).first()
    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401

    access_token = create_access_token(identity=username)
    return jsonify({'access_token': access_token})

# Text processing routes
@app.route('/generate/text', methods=['POST'])
@jwt_required()
@limiter.limit("10/minute")
def generate_text():
    """
    Generate text using DeepSeek
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          $ref: '#/definitions/TextRequest'
    responses:
      200:
        description: Generated text
    """
    data = request.json
    prompt = data.get('text')
    max_length = data.get('max_length', 100)

    try:
        response = code_generator(
            prompt,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7
        )
        return jsonify({'generated_text': response[0]['generated_text']})
    except Exception as e:
        logging.error(f"Text generation error: {str(e)}")
        return jsonify({'error': 'Text generation failed'}), 500

@app.route('/generate/code', methods=['POST'])
@jwt_required()
@limiter.limit("10/minute")
def generate_code():
    """
    Generate code using DeepSeek Coder
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          $ref: '#/definitions/CodeRequest'
    responses:
      200:
        description: Generated code
    """
    data = request.json
    prompt = data.get('prompt')
    max_length = data.get('max_length', 200)

    try:
        response = code_generator(
            f"<｜begin▁of▁sentence｜>{prompt}<｜end▁of▁sentence｜>",
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7
        )
        return jsonify({'generated_code': response[0]['generated_text']})
    except Exception as e:
        logging.error(f"Code generation error: {str(e)}")
        return jsonify({'error': 'Code generation failed'}), 500

@app.route('/analyze/sentiment', methods=['POST'])
@jwt_required()
def analyze_sentiment():
    """
    Analyze text sentiment
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          $ref: '#/definitions/TextRequest'
    responses:
      200:
        description: Sentiment analysis result
    """
    data = request.json
    text = data.get('text')

    try:
        result = sentiment_analyzer(text)
        return jsonify({'sentiment': result[0]})
    except Exception as e:
        logging.error(f"Sentiment analysis error: {str(e)}")
        return jsonify({'error': 'Sentiment analysis failed'}), 500

# File processing routes
@app.route('/upload', methods=['POST'])
@jwt_required()
def upload_file():
    """
    Upload and process file
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      - name: description
        in: formData
        type: string
    responses:
      200:
        description: File processed successfully
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            if filename.endswith('.pdf'):
                text = ''
                with open(file_path, 'rb') as f:
                    pdf = PdfReader(f)
                    for page in pdf.pages:
                        text += page.extract_text()
            else:
                with open(file_path, 'r') as f:
                    text = f.read()

            user = User.query.filter_by(username=get_jwt_identity()).first()
            saved_text = SavedText(
                user_id=user.id,
                content=text,
                file_path=file_path
            )
            db.session.add(saved_text)
            db.session.commit()

            return jsonify({
                'message': 'File uploaded and processed successfully',
                'text': text[:500] + '...' if len(text) > 500 else text
            })
        except Exception as e:
            logging.error(f"File processing error: {str(e)}")
            return jsonify({'error': 'File processing failed'}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

# Additional NLP routes
@app.route('/translate', methods=['POST'])
@jwt_required()
def translate_text():
    """
    Translate text
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          $ref: '#/definitions/TranslationRequest'
    responses:
      200:
        description: Translated text
    """
    data = request.json
    text = data.get('text')
    target_lang = data.get('target_lang', 'en')

    try:
        result = translator(text, tgt_lang=target_lang)
        return jsonify({'translated_text': result[0]['translation_text']})
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return jsonify({'error': 'Translation failed'}), 500

@app.route('/analyze/ner', methods=['POST'])
@jwt_required()
def analyze_ner():
    """
    Named Entity Recognition
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          $ref: '#/definitions/TextRequest'
    responses:
      200:
        description: NER results
    """
    data = request.json
    text = data.get('text')

    try:
        result = ner_pipeline(text)
        return jsonify({'entities': result})
    except Exception as e:
        logging.error(f"NER error: {str(e)}")
        return jsonify({'error': 'NER analysis failed'}), 500

@app.route('/correct/grammar', methods=['POST'])
@jwt_required()
def correct_grammar():
    """
    Grammar correction
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          $ref: '#/definitions/TextRequest'
    responses:
      200:
        description: Corrected text
    """
    data = request.json
    text = data.get('text')

    try:
        result = grammar_checker(f"grammar: {text}")
        return jsonify({'corrected_text': result[0]['generated_text']})
    except Exception as e:
        logging.error(f"Grammar correction error: {str(e)}")
        return jsonify({'error': 'Grammar correction failed'}), 500

# Admin routes
@app.route('/admin/users', methods=['GET'])
@jwt_required()
@admin_required
def list_users():
    """
    List all users (Admin only)
    ---
    responses:
      200:
        description: List of users
    """
    users = User.query.all()
    return jsonify([{
        'id': user.id,
        'username': user.username,
        'is_admin': user.is_admin
    } for user in users])

@app.route('/admin/promote/<username>', methods=['POST'])
@jwt_required()
@admin_required
def promote_user(username):
    """
    Promote user to admin (Admin only)
    ---
    parameters:
      - name: username
        in: path
        type: string
        required: true
    responses:
      200:
        description: User promoted
    """
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404

    user.is_admin = True
    db.session.commit()
    return jsonify({'message': 'User promoted to admin'})

# Cache management
@app.route('/cache/clear', methods=['POST'])
@jwt_required()
@admin_required
def clear_cache():
    """
    Clear application cache (Admin only)
    ---
    responses:
      200:
        description: Cache cleared
    """
    cache.clear()
    return jsonify({'message': 'Cache cleared successfully'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=os.getenv('DEBUG') == 'true')
