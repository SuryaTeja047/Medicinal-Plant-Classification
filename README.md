# Image Classification Project

## Overview
This project is an image classification system that categorizes images into 40 different classes. The project involves data splitting, model training, and a Streamlit app for showcasing the results.
Download Dataset and add to the main folder and name the folder as data.
[Download Dataset](https://www.kaggle.com/datasets/warcoder/indian-medicinal-plant-image-dataset) 

## Project Structure
- `data/`: Directory containing image datasets organized into 40 directories (one for each class) Data set Link.
- `app.py`: Streamlit app for displaying classification results.
- `requirements.txt`: List of required Python packages.


```bash

pip install -r requirements.txt

python data_split.py

python train_model.py

streamlit run app.py
