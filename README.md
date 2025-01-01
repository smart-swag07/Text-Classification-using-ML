# Text Classification Using Machine Learning  

This project demonstrates a machine learning-based system for text classification, categorizing textual data into predefined classes like news topics. It uses models such as Naive Bayes and SVM, alongside techniques like TF-IDF for feature extraction.  

---

## Features  
- Machine learning models: Naive Bayes, Complement Naive Bayes, and SVM.  
- Text preprocessing: tokenization, stopword removal, and TF-IDF vectorization.  
- Flask-based web interface for real-time text classification.  

---

## How to Run the Project

1. Clone the Repository
git clone <repository_url>
cd <repository_folder>

2. Prepare the Dataset
Place the dataset (bbc-text.csv) in the project directory.
Ensure the dataset has two columns: Text and Category.

3. Train the Models
Train the machine learning models by running:
python model.py

4. Start the Application
Launch the web interface:
python app.py

5. Access the Application
Open your browser and go to:
http://127.0.0.1:5000

6. Enter text to classify it into predefined categories.

