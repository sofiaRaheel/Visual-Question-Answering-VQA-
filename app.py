# import os
# import torch
# from PIL import Image
# from flask import Flask, request, jsonify, render_template
# from transformers import BlipProcessor, BlipForQuestionAnswering

# app = Flask(__name__)

# def load_model():
#     try:
#         # Download and load the model from Hugging Face Hub
#         model_name = "Salesforce/blip-vqa-base"
#         processor = BlipProcessor.from_pretrained(model_name)
#         model = BlipForQuestionAnswering.from_pretrained(model_name).to('cpu')
#         print("Model loaded successfully from Hugging Face Hub!")
#         return processor, model
#     except Exception as e:
#         print(f"Model loading failed: {str(e)}")
#         return None, None

# processor, model = load_model()

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/analyze', methods=['POST'])
# def analyze():
#     if model is None:
#         return jsonify({'error': 'Model not loaded'}), 500

#     try:
#         image = Image.open(request.files['image'].stream)
#         question = request.form.get('question', '').strip()
        
#         if not question:
#             return jsonify({'error': 'No question provided'}), 400

#         inputs = processor(images=image, text=question, return_tensors="pt")
#         outputs = model.generate(**inputs)
#         answer = processor.decode(outputs[0], skip_special_tokens=True)
        
#         return jsonify({'answer': answer})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

# import os
# import torch
# from PIL import Image
# from flask import Flask, request, jsonify, render_template
# from transformers import BlipProcessor, BlipForQuestionAnswering
# from torch.optim import AdamW
# from torch.utils.data import Dataset, DataLoader
# import json
# from datetime import datetime
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# # Configuration
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
# app.config['MODEL_CACHE'] = 'model_cache'
# app.config['FINE_TUNE_DATA'] = 'fine_tune_data.json'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# os.makedirs(app.config['MODEL_CACHE'], exist_ok=True)

# # Global variables
# processor = None
# model = None
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def load_model(use_cache=True):
#     global processor, model
    
#     try:
#         model_name = "Salesforce/blip-vqa-base"
#         cache_path = os.path.join(app.config['MODEL_CACHE'], model_name.replace('/', '_'))
        
#         # Load from cache if available
#         if use_cache and os.path.exists(cache_path):
#             print(f"Loading model from cache: {cache_path}")
#             processor = BlipProcessor.from_pretrained(cache_path)
#             model = BlipForQuestionAnswering.from_pretrained(cache_path).to(device)
#         else:
#             print("Downloading model from Hugging Face Hub...")
#             processor = BlipProcessor.from_pretrained(model_name)
#             model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)
            
#             # Save to cache
#             model.save_pretrained(cache_path)
#             processor.save_pretrained(cache_path)
            
#         print(f"Model loaded successfully on device: {device}")
#         return True
#     except Exception as e:
#         print(f"Model loading failed: {str(e)}")
#         return False

# # Custom dataset for fine-tuning
# class VQADataset(Dataset):
#     def __init__(self, data, processor):
#         self.data = data
#         self.processor = processor

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         image = Image.open(item['image_path']).convert('RGB')
#         text = item['question']
#         answer = item['answer']
        
#         encoding = self.processor(
#             images=image, 
#             text=text, 
#             padding="max_length", 
#             return_tensors="pt",
#             truncation=True
#         )
        
#         # For generation task, we need to prepare labels
#         labels = self.processor.tokenizer(
#             answer,
#             padding="max_length", 
#             return_tensors="pt",
#             truncation=True
#         ).input_ids
        
#         encoding["labels"] = labels
#         return {k: v.squeeze() for k, v in encoding.items()}

# # Initialize model at startup
# if not load_model():
#     print("Failed to initialize model. Some features may not work.")

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/analyze', methods=['POST'])
# def analyze():
#     if model is None:
#         return jsonify({'error': 'Model not loaded'}), 500

#     try:
#         # Check if image was uploaded
#         if 'image' not in request.files:
#             return jsonify({'error': 'No image uploaded'}), 400
        
#         file = request.files['image']
#         if file.filename == '':
#             return jsonify({'error': 'No selected file'}), 400
            
#         if not allowed_file(file.filename):
#             return jsonify({'error': 'Invalid file type'}), 400
            
#         # Save the file temporarily
#         filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         # Process the image
#         image = Image.open(filepath).convert('RGB')
#         question = request.form.get('question', '').strip()
        
#         if not question:
#             os.remove(filepath)
#             return jsonify({'error': 'No question provided'}), 400

#         inputs = processor(images=image, text=question, return_tensors="pt").to(device)
#         outputs = model.generate(**inputs)
#         answer = processor.decode(outputs[0], skip_special_tokens=True)
        
#         # Clean up
#         os.remove(filepath)
        
#         return jsonify({
#             'answer': answer,
#             'question': question,
#             'image': filename
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/fine-tune', methods=['POST'])
# def fine_tune():
#     if model is None:
#         return jsonify({'error': 'Model not loaded'}), 500
        
#     try:
#         # Check if training data was provided
#         if 'file' not in request.files:
#             return jsonify({'error': 'No training data uploaded'}), 400
            
#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({'error': 'No selected file'}), 400
            
#         if not file.filename.endswith('.json'):
#             return jsonify({'error': 'Training data must be JSON'}), 400
            
#         # Save training data
#         filename = secure_filename(f"train_{datetime.now().timestamp()}.json")
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         # Load training data
#         with open(filepath, 'r') as f:
#             train_data = json.load(f)
            
#         # Validate training data format
#         required_keys = {'image_path', 'question', 'answer'}
#         for item in train_data:
#             if not all(k in item for k in required_keys):
#                 os.remove(filepath)
#                 return jsonify({'error': f'Invalid training data format. Required keys: {required_keys}'}), 400
        
#         # Create dataset and dataloader
#         dataset = VQADataset(train_data, processor)
#         dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
#         # Training setup
#         optimizer = AdamW(model.parameters(), lr=5e-5)
#         model.train()
        
#         # Training loop (simplified)
#         for batch in dataloader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             outputs = model(**batch)
#             loss = outputs.loss
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
        
#         # Save fine-tuned model
#         model.save_pretrained(app.config['MODEL_CACHE'])
#         processor.save_pretrained(app.config['MODEL_CACHE'])
        
#         # Clean up
#         os.remove(filepath)
        
#         return jsonify({'message': 'Model fine-tuning completed successfully'})
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/upload-training-data', methods=['POST'])
# def upload_training_data():
#     try:
#         data = request.json
#         if not data or not isinstance(data, list):
#             return jsonify({'error': 'Invalid data format. Expected a list of training examples.'}), 400
            
#         # Validate each example
#         required_keys = {'image_path', 'question', 'answer'}
#         for example in data:
#             if not all(k in example for k in required_keys):
#                 return jsonify({'error': f'Each example must contain: {required_keys}'}), 400
        
#         # Save to training data file
#         with open(app.config['FINE_TUNE_DATA'], 'w') as f:
#             json.dump(data, f)
            
#         return jsonify({'message': f'Successfully saved {len(data)} training examples'})
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)

import os
import torch
import re
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template
from transformers import BlipProcessor, BlipForQuestionAnswering
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MODEL_CACHE'] = 'model_cache'
app.config['FINE_TUNE_DATA'] = 'fine_tune_data.json'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['MAX_QUESTION_LENGTH'] = 512  # Characters
app.config['IMAGE_SIZE'] = (384, 384)  # Target size for preprocessing

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_CACHE'], exist_ok=True)

# Global variables
processor = None
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def clean_text(text):
    """Enhanced text preprocessing"""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
    text = text[:app.config['MAX_QUESTION_LENGTH']]  # Truncate
    return text

def preprocess_image(image):
    """Enhanced image preprocessing"""
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize with aspect ratio preservation
    image.thumbnail(app.config['IMAGE_SIZE'])
    
    # Pad to exact size if needed
    delta_w = app.config['IMAGE_SIZE'][0] - image.size[0]
    delta_h = app.config['IMAGE_SIZE'][1] - image.size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    image = ImageOps.expand(image, padding, fill='black')
    
    return image

def load_model(use_cache=True):
    global processor, model
    
    try:
        model_name = "Salesforce/blip-vqa-base"
        cache_path = os.path.join(app.config['MODEL_CACHE'], model_name.replace('/', '_'))
        
        if use_cache and os.path.exists(cache_path):
            print(f"Loading model from cache: {cache_path}")
            processor = BlipProcessor.from_pretrained(cache_path)
            model = BlipForQuestionAnswering.from_pretrained(cache_path).to(device)
        else:
            print("Downloading model from Hugging Face Hub...")
            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)
            
            model.save_pretrained(cache_path)
            processor.save_pretrained(cache_path)
            
        print(f"Model loaded successfully on device: {device}")
        return True
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        return False

class VQADataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = preprocess_image(Image.open(item['image_path']))
        text = clean_text(item['question'])
        answer = clean_text(item['answer'])
        
        encoding = self.processor(
            images=image, 
            text=text, 
            padding="max_length", 
            return_tensors="pt",
            truncation=True,
            max_length=64  # For both question and answer
        )
        
        labels = self.processor.tokenizer(
            answer,
            padding="max_length", 
            return_tensors="pt",
            truncation=True,
            max_length=64
        ).input_ids
        
        encoding["labels"] = labels
        return {k: v.squeeze() for k, v in encoding.items()}

# Initialize model
if not load_model():
    print("Failed to initialize model. Some features may not work.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
            
        filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        image = preprocess_image(Image.open(filepath))
        question = clean_text(request.form.get('question', ''))
        
        if not question:
            os.remove(filepath)
            return jsonify({'error': 'No question provided'}), 400

        inputs = processor(
            images=image, 
            text=question, 
            return_tensors="pt",
            truncation=True,
            max_length=64
        ).to(device)
        
        outputs = model.generate(**inputs)
        answer = processor.decode(outputs[0], skip_special_tokens=True)
        
        os.remove(filepath)
        
        return jsonify({
            'answer': answer,
            'question': question,
            'image': filename,
            'processing_details': {
                'image_size': app.config['IMAGE_SIZE'],
                'max_question_length': app.config['MAX_QUESTION_LENGTH']
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fine-tune', methods=['POST'])
def fine_tune():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No training data uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not file.filename.endswith('.json'):
            return jsonify({'error': 'Training data must be JSON'}), 400
            
        filename = secure_filename(f"train_{datetime.now().timestamp()}.json")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        with open(filepath, 'r') as f:
            train_data = json.load(f)
            
        required_keys = {'image_path', 'question', 'answer'}
        for item in train_data:
            if not all(k in item for k in required_keys):
                os.remove(filepath)
                return jsonify({'error': f'Invalid training data format. Required keys: {required_keys}'}), 400
        
        dataset = VQADataset(train_data, processor)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        optimizer = AdamW(model.parameters(), lr=5e-5)
        model.train()
        
        for epoch in range(3):  # 3 epochs
            total_loss = 0
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")
        
        model.save_pretrained(app.config['MODEL_CACHE'])
        processor.save_pretrained(app.config['MODEL_CACHE'])
        
        os.remove(filepath)
        
        return jsonify({
            'message': 'Fine-tuning completed',
            'epochs': 3,
            'average_loss': total_loss/len(dataloader)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)