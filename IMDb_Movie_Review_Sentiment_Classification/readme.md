# IMDb Movie Review Sentiment Analysis

An end-to-end NLP and machine learning project for classifying IMDb movie reviews as **Positive** or **Negative**.

### Key Components
- Exploratory data analysis of 50,000 IMDb reviews
- Text cleaning, tokenization, stopword removal, and lemmatization
- Bag of Words and TF-IDF feature extraction
- Multinomial Naive Bayes, Logistic Regression, and Linear SVM
- Evaluation using Accuracy, Precision, Recall, F1-score, and Confusion Matrix
- Prediction on unseen movie reviews

### Results
After removing duplicate reviews, the dataset contained **49,582 reviews**. The **Linear SVM with TF-IDF features** achieved the best overall performance:

- **Accuracy:** 90.27%
- **Precision:** 89.83%
- **Recall:** 90.90%
- **F1-score:** 90.36%

The project demonstrates a complete NLP classification workflow, from raw text preprocessing and feature engineering to model comparison and prediction.
