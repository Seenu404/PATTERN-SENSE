
# Pattern Sense - Project Documentation

## 1. Introduction

- **Project Title:** Pattern Sense  
- **Team Members:**  
  - A. Chinna Seenu
  - G. Madhavi
  - P. Ganga Lakshmi
  - K. Blessy Sterlin
  - Bhanu Priya

## 2. Project Overview

- **Purpose:**  
  Pattern Sense is an AI-powered web application designed to classify fabric patterns using deep learning. The goal is to automate the identification of various fabric designs to assist textile industries, designers, or e-commerce platforms.

- **Features:**  
  - Upload fabric pattern images  
  - Classifies images using a trained CNN model  
  - Web interface for user interaction  
  - Real-time prediction result display

## 3. Architecture

- **Model (AI Component):**  
  - Deep Learning CNN model built using TensorFlow/Keras  
  - Pre-trained on a custom dataset of fabric patterns

- **Web Backend:**  
  - Flask (Python) handles server-side logic  
  - Loads the trained model and performs predictions  
  - Image preprocessing with OpenCV

- **Frontend:**  
  - HTML, CSS, Bootstrap for styling  
  - User can upload images and view predictions

- **No Database:**  
  - This project does not use a persistent database as it is inference-only.

## 4. Setup Instructions

- **Prerequisites:**  
  - Python 3.7+  
  - pip  
  - TensorFlow  
  - Flask  
  - OpenCV  
  - Matplotlib

- **Installation:**  
  1. Clone the repo:  
     ```bash
     git clone https://github.com/Seenu404/PATTERN-SENSE.git
     cd PATTERN-SENSE
     ```
  2. Install dependencies:  
     ```bash
     pip install -r requirements.txt
     ```
  3. Place the model file (`model.h5`) in the project root (or update the path accordingly).
  4. Run the application:  
     ```bash
     python app.py
     ```

## 5. Folder Structure

```
PATTERN-SENSE/
│
├── static/             # CSS and asset files
│   └── assets/         # Images and background
│
├── templates/          # HTML files (home, predict, result pages)
│
├── model/              # Folder for .h5 model (optional)
│
├── app.py              # Main Flask application
├── requirements.txt    # List of dependencies
└── README.md
```

## 6. Running the Application

- Start the Flask app with:
  ```bash
  python app.py
  ```
- Navigate to `http://localhost:5000` in your browser to interact with the application.

## 7. API Documentation

| Endpoint   | Method | Description                 |
|------------|--------|-----------------------------|
| `/`        | GET    | Loads home page             |
| `/predict` | GET    | Opens prediction form       |
| `/predict` | POST   | Accepts uploaded image, runs prediction, returns result |

- **Example Request:**
  ```bash
  POST /predict
  Content-Type: multipart/form-data
  Body: image file
  ```

- **Example Response:**
  ```json
  {
    "prediction": "Floral Pattern"
  }
  ```

## 8. Authentication

- No authentication is used in this project, as it is a publicly accessible demonstration tool.

## 9. User Interface

- Simple, aesthetic UI with light theme  
- Main pages:
  - Home: Welcome page with project info  
  - Predict: Upload image page  
  - Result: Displays prediction result  

- **Screenshots:** *(Add if available)*

## 10. Testing

- Manual testing done via:
  - Uploading multiple pattern types
  - Checking model prediction accuracy
  - Verifying user interface functionality

## 11. Screenshots or Demo

  ![Project1](https://github.com/user-attachments/assets/17bf57dd-4f20-44a9-b874-b722a7941291)

-Demo Link : https://youtu.be/5Il8xABnexY?si=QE5-rmCR7C_D-HOe

## 12. Known Issues

- Limited to image resolution and quality
- Model trained on a relatively small dataset
- No database logging or history of predictions

## 13. Future Enhancements

- Train model on larger and diverse datasets
- Add pattern sub-categories
- Include a feedback mechanism for model predictions
- Deploy on cloud (e.g., Render, Heroku, or AWS)
- Add database to log user inputs and results
