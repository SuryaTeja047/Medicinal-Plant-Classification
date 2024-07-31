import streamlit as st
import numpy as np
from classify import classify_image

# Load class labels (ensure this matches your trained model's labels)
class_labels = [
    'Aloevera', 'Amla', 'Amruta Balli', 'Arali', 'Ashoka', 'Ashwagandha',
    'Avocado', 'Bamboo', 'Basale', 'Betel', 'Betel Nut', 'Brahmi', 'Castor',
    'Curry Leaf', 'Doddapatre', 'Ekka', 'Ganike', 'Guava', 'Geranium',
    'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jasmine', 'Lemon', 'Lemon Grass',
    'Mango', 'Mint', 'Nagadali', 'Neem', 'Nithyapushpa', 'Nooni', 'Papaya',
    'Pepper', 'Pomegranate', 'Raktachandini', 'Rose', 'Sapota', 'Tulasi',
    'Wood Sorel'
]

# Plant information dictionary
plant_info = {
    'Aloevera': 'Aloe vera is renowned for its soothing properties and is commonly used in skincare to heal and hydrate.',
    'Amla': 'Amla, or Indian gooseberry, is rich in Vitamin C and is used to boost immunity and improve skin health.',
    'Amruta Balli': 'Amruta Balli is noted for its anti-inflammatory and antioxidant benefits, supporting overall health.',
    'Arali': 'Arali has traditional use in relieving pain and is known for its analgesic properties.',
    'Ashoka': 'Ashoka is often used to alleviate menstrual discomfort and improve skin complexion.',
    'Ashwagandha': 'Ashwagandha is known for its stress-relieving properties and its ability to enhance vitality and reduce anxiety.',
    'Avocado': 'Avocado is rich in healthy fats and is beneficial for heart health and skin hydration.',
    'Bamboo': 'Bamboo is used in traditional medicine for its anti-inflammatory properties, supporting joint and muscle health.',
    'Basale': 'Basale is known for its nutritional value and is used to treat respiratory issues and support general health.',
    'Betel': 'Betel is chewed for its stimulant effects and has cultural significance in many traditional practices.',
    'Betel Nut': 'Betel Nut is used as a stimulant and in various traditional ceremonies, also known for its digestive benefits.',
    'Brahmi': 'Brahmi is known for enhancing cognitive functions and alleviating stress, supporting mental clarity and relaxation.',
    'Castor': 'Castor oil is used for its laxative effects and in skincare for its moisturizing and healing properties.',
    'Curry Leaf': 'Curry Leaf is rich in antioxidants and is used in culinary and medicinal applications for digestive health.',
    'Doddapatre': 'Doddapatre is valued for its antiseptic properties and is used to treat respiratory issues and infections.',
    'Ekka': 'Ekka is recognized for its diverse medicinal properties, aiding in the treatment of various ailments.',
    'Ganike': 'Ganike has traditional uses for its healing properties, supporting overall well-being.',
    'Guava': 'Guava is packed with Vitamin C and supports digestion, immunity, and skin health.',
    'Geranium': 'Geranium is used in aromatherapy for its calming effects and in skincare for its soothing properties.',
    'Henna': 'Henna is known for its natural dyeing capabilities and cooling effects on the skin.',
    'Hibiscus': 'Hibiscus is used to lower blood pressure and improve skin health, also known for its antioxidant properties.',
    'Honge': 'Honge is valued for its traditional medicinal uses, supporting various health benefits.',
    'Insulin': 'Insulin is used for managing diabetes and is known for its role in regulating blood sugar levels.',
    'Jasmine': 'Jasmine is used in aromatherapy for its calming scent and in skincare for its soothing effects.',
    'Lemon': 'Lemon is rich in Vitamin C and is used for detoxification and enhancing overall health.',
    'Lemon Grass': 'Lemon Grass is used in cooking and traditional medicine for its digestive and anti-inflammatory benefits.',
    'Mango': 'Mango is rich in vitamins and is used for its nutritional benefits and support for overall health.',
    'Mint': 'Mint is known for its cooling effects and is used to aid digestion and freshen breath.',
    'Nagadali': 'Nagadali has traditional uses in healing and supporting various health conditions.',
    'Neem': 'Neem is known for its antibacterial and antifungal properties, supporting skin health and overall hygiene.',
    'Nithyapushpa': 'Nithyapushpa is used in traditional remedies for its wide range of medicinal benefits.',
    'Nooni': 'Nooni is recognized for its health benefits and traditional medicinal uses in various treatments.',
    'Papaya': 'Papaya is rich in digestive enzymes and supports digestion, skin health, and overall wellness.',
    'Pepper': 'Pepper is used both as a spice and for its medicinal properties, including aiding digestion and respiratory health.',
    'Pomegranate': 'Pomegranate is rich in antioxidants and is beneficial for heart health and overall vitality.',
    'Raktachandini': 'Raktachandini is used in traditional medicine for its healing properties and overall health support.',
    'Rose': 'Rose is known for its soothing effects and is used in skincare and aromatherapy.',
    'Sapota': 'Sapota is rich in vitamins and is used for its nutritional benefits and overall health support.',
    'Tulasi': 'Tulasi, or holy basil, is known for its immune-boosting properties and is used to enhance overall wellness.',
    'Wood Sorel': 'Wood Sorel is used in traditional medicine for its medicinal properties and health benefits.'
}

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Plant Identification"])

# Home Page
if app_mode == "Home":
    st.title("Welcome to Plant Sense")
    st.image("images/   homepage.jpg", use_column_width=True)

    st.markdown("""
    **Plant Sense** offers a powerful solution for identifying medicinal plants using advanced machine learning techniques.

    ### How It Works
    1. **Upload Image:** Go to the **Plant Identification** page to upload a plant image.
    2. **Analysis:** Our model will analyze the image to determine the plant species and its potential uses.
    3. **Results:** View detailed results and learn about the plant's medicinal properties.

    ### Why Choose Plant Sense?
    - **Precision:** Our CNN-based model ensures high accuracy in plant identification.
    - **Ease of Use:** Intuitive interface designed for a seamless experience.
    - **Rapid Results:** Get instant feedback and recommendations.

    ### Explore More
    Visit the **Plant Identification** page to start using our system!
    """)

# About Page
elif app_mode == "About":
    st.title("About This Project")
    st.markdown("""
    ### Dataset Overview
    This dataset includes around 90,000 RGB images of medicinal plants, categorized into 40 species. Enhanced through offline augmentation techniques, the dataset is divided into training and validation sets (80/20) and includes a separate directory for test images.

    ### Technologies Used
    - **Convolutional Neural Networks (CNNs):** For advanced image classification.
    - **TensorFlow:** Framework for model training and evaluation.
    - **Streamlit:** For building an interactive and user-friendly interface.
    - **NumPy:** For numerical operations and data handling.
    - **Pandas:** For preprocessing and managing data.
    - **Matplotlib:** For visualizing model performance and results.

    This project aims to support accurate and efficient medicinal plant identification using state-of-the-art technology.
    """)

# Prediction Page
elif app_mode == "Plant Identification":
    st.title("Plant Identification")
    st.markdown("""
    Upload an image of a medicinal plant to get its identification and learn about its uses.

    ### Instructions
    1. Click **Browse files** to upload a plant image.
    2. Click **Show Image** to preview the uploaded image.
    3. Click **Predict** to receive the plant's identification and information.
    """)

    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    # Display the uploaded image
    if st.button("Show Image"):
        if test_image is not None:
            st.image(test_image, caption="Uploaded Image", use_column_width=True)
        else:
            st.warning("Please upload an image first.")

    # Prediction button
    if st.button("Predict"):
        if test_image is not None:
            # st.snow()  # Show a spinner while processing
            st.write("Analyzing the image...")
            
            # Save the uploaded image to a temporary file
            temp_image_path = "temp_image.jpg"
            with open(temp_image_path, "wb") as temp_file:
                temp_file.write(test_image.getbuffer())

            # Get classification results
            predictions = classify_image(temp_image_path)
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_labels[predicted_class_index]

            # Display result
            st.success(f"Prediction: {predicted_class}")
            st.write("### Plant Description and Uses")
            st.write(plant_info.get(predicted_class, "Description not available."))

            # Uncomment the following lines if you want to display the confidence scores
            # st.write("Confidence Scores:")
            # for i, (label, score) in enumerate(zip(class_labels, predictions[0])):
            #     st.write(f"{label}: {score * 100:.2f}%")
        else:
            st.warning("Please upload an image first.")
