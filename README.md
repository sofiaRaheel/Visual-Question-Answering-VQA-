# Visual Question Answering (VQA) Web App

A web-based mini-project that enables users to upload an image and ask questions about it. The app responds with an answer using a pre-trained **BLIP VQA model**, combining computer vision and natural language processing for interactive image understanding.

---

## Project Overview

- **Focus**: Image Understanding + Vision-Language Modeling
 This project demonstrates a simple but powerful VQA system built with:
- Pretrained BLIP model for Vision-Language understanding
- Flask for the web backend
- PIL for image processing
- Optional fine-tuning functionality on custom training data

---

## Features

* Upload an image  
* Ask a question about the image  
* Get an AI-generated answer  
* Preprocess images and questions for better model performance  
* Optional: Fine-tune the model on your custom dataset  

---

---


##  How It Works

###  Image Upload & Question Input
- User uploads an image and submits a related question via a simple web form (`index.html`).
- The backend receives this input and starts preprocessing.

###  Image Preprocessing
- Converts to RGB if needed.
- Resizes the image to 384x384, preserving aspect ratio and padding with black borders.

###  Question Preprocessing
- Strips whitespace, limits length to 512 characters, and normalizes spacing.

###  VQA Model Inference
- The `BlipProcessor` tokenizes the image and question.
- The `BlipForQuestionAnswering` model generates an answer using `generate()`.

###  Answer Response
- The answer is decoded and returned to the user.
- Temporary image file is deleted after use.

