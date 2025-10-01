 🌍 Disaster Tweet Classifier

This project is a Natural Language Processing (NLP) machine learning model that classifies tweets as *disaster-related* or *not disaster-related*.  
It demonstrates text pre-processing, feature engineering, and classification using *Python & Scikit-learn*.

---

📂 Project Structure
disaster_tweet_classifier/
│── classifier.ipynb   # Jupyter Notebook (step-by-step implementation)
│── classifier.py      # Python script version
│── train.csv          # Training dataset (from Kaggle)
│── README.md          # Project documentation
│── requirements.txt   # Python dependencies


⚙️ Technologies Used
- Python 3  
- Pandas & NumPy  
- Scikit-learn  
- NLTK  

---

🚀 How to Run
1. Clone this repository:
   git clone https://github.com/riabarnes/Disaster-Tweet-Classifier.git
   cd Disaster-Tweet-Classifier

2.	Install the required libraries:
pip install -r requirements.txt

3.	Run the notebook:
jupyter notebook classifier.ipynb

OR run the Python script:
python classifier.py

Results
	•	Model: Logistic Regression
	•	Accuracy: ~80% on validation set
	•	Evaluated using precision, recall, and F1-score


📈 Future Improvements
	•	Try deep learning models like LSTM / BERT
	•	Hyperparameter tuning for better performance
	•	Deploy as a web app with Flask or Streamlit


📑 Dataset

Dataset: Kaggle NLP Disaster Tweets Competition
(Please download train.csv from Kaggle if not included in this repo.)


✨ Author

Created by Ria Barnes ✨
