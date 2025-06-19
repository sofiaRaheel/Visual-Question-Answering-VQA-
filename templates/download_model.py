from transformers import BlipProcessor, BlipForQuestionAnswering
import os

# Set cache location
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_cache')

print("Downloading BLIP model (this may take a few minutes)...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
print("Model downloaded successfully to:", os.environ['HF_HOME'])