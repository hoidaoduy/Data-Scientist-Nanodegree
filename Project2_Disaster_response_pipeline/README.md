# Disaster Response Pipeline Project
### Project Overview

This project builds a Natural Language Processing (NLP) model to categorize messages sent during disasters. The goal is to help emergency response organizations quickly identify and route messages to appropriate disaster relief agencies.

The project includes:
- ETL Pipeline to clean and process message data
- Machine Learning Pipeline to train a multi-output classifier
- Web App to classify new messages in real-time

The web application allows emergency workers to input new messages and get classification results across 36 different categories. It also displays visualizations of the training dataset.

Key features:
- Uses scikit-learn's machine learning pipeline
- Implements multi-label classification
- Built with Flask web framework
- Interactive visualizations using Plotly
- Bootstrap-based responsive UI

### Demo Screenshots
#### Main Page
![Main Page](screencapture-127-0-0-1-3001-2024-10-30-14_02_02.png)

#### Classification Results
![Classification Results](screencapture-127-0-0-1-3001-go-2024-10-30-14_04_00.png)

### Project Structure
- app
    - templates
        - master.html               # Main page
        - go.html                   # Classification results page
    - run.py                        # Flask application

- data
    - disaster_messages.csv         # Original dataset
    - disaster_categories.csv       # Classification labels
    - DisasterResponse.db           # SQLite database
    - process_data.py               # ETL pipeline script

- models
    - classifier.pkl                # Trained model
    - train_classifier.py           # ML pipeline script

- README.md                         # Project documentation


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
