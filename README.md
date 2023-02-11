# Hate Speech Analysis on Twitter Dataset
# Overview
This repository is a machine learning model for detecting hate speech on Twitter. The dataset consists of tweets that are labeled as either hate speech, offensive language, or neither.

The purpose of this model is to help identify and prevent hateful or discriminatory language on social media platforms. It can be used as a tool for moderators or analysts to track and flag harmful content.This is using to classify tweets having hate speech from others.

# Getting Started
## Prerequisites
- Python 3
- Jupyter Notebook
- Pandas
- Scikit-learn
- NLTK
- Seaborn
- Tf-Idf
## Installation
1. Clone the repository to your local machine.
2. Install Python 3, Jupyter Notebook, Pandas, Scikit-learn, NLTK, and Matplotlib.
3. Open the notebook '**hate_speech_analysis.ipynb**' in Jupyter Notebook.
4. Run each cell in the notebook to load the data, preprocess the text, train and test the model, and analyze the results.
## Dataset
- The **Train** dataset contain **31962 Rows** and having 3 columns **['id','label','tweet']**.
- The **Test** dataset contain **17197 Rows** and having 2 columns **['id','tweet']**.
## Preprocessing
The text in the tweets is preprocessed using NLTK. The preprocessing steps include removing @ word,numbers,Greek Characters,hmm and it's forms,slang words,stop words, punctuation, and finding the # attached words.
## Model
In trainied 6 models on the given Trained dataset
1. LogisticRegression
2. Navie Bayes
3. Random Forest
4. SGD classifier
5. SVM classfier
6. GradientBoosting

**Logistic Regresssion** gives good accuaracy over all models.

The model used in this analysis is a **Logistic Regression**. The model is trained on the preprocessed text and the labels in the dataset.

The performance of the model is evaluated using precision, recall, and F1-score. The results are displayed in a confusion matrix and a classification report.
## Results
The **Logistic Regression** model achieves an F1-score of 0.77 for identifying hate speech.**final_results.csv** having the final results of given test dataset.
## Conclusion
This machine learning model can be used to automatically detect hate speech and offensive language on Twitter. It can be a valuable tool for moderators or analysts to track and flag harmful content like sexism and racisim contents. However, the model is not perfect and may have limitations when dealing with different languages or dialects.
