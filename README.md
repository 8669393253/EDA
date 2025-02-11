# EDA
Sentiment Analysis on tweets using NLP and ML techniques. It processes textual data, extracts features using TF-IDF, and classifies tweets as positive or negative using a Multinomial Naïve Bayes model. The analysis also includes EDA with visualizations like word clouds and sentiment distribution graphs.

# Twitter Sentiment Analysis

## Overview
This project performs **Sentiment Analysis** on tweets using **Natural Language Processing (NLP)** and **Machine Learning** techniques. It processes textual data, extracts features using **TF-IDF**, and classifies tweets as **positive** or **negative** using a **Multinomial Naïve Bayes** model. The analysis also includes **Exploratory Data Analysis (EDA)** with **visualizations** like word clouds and sentiment distribution graphs.

## Features
- **Data Preprocessing**: Cleans and normalizes textual data by removing unnecessary elements like mentions, hashtags, links, and special characters.
- **Exploratory Data Analysis (EDA)**: Provides insights into sentiment distribution and common words using visualizations.
- **TextBlob Sentiment Analysis**: Uses TextBlob to assess the polarity of text and classify it as positive or negative.
- **Machine Learning Model**: Implements a Naïve Bayes classifier for sentiment classification after feature extraction using TF-IDF.
- **Evaluation Metrics**: Measures model performance using accuracy, classification reports, and confusion matrices.

## Installation
Ensure you have **Python 3.x** installed and install the required dependencies using package managers like pip.

## Dataset
The dataset is sourced from an online repository and consists of labeled tweets. Sentiments are categorized as:
- **0** → Negative
- **1** → Positive

After loading the dataset, sentiment values are mapped to meaningful labels (negative/positive) for easier interpretation.

## Workflow
1. **Load Dataset**: The dataset is imported and structured for analysis.
2. **Preprocess Data**: Text is cleaned by removing special characters, links, and non-alphabetic elements.
3. **Perform Exploratory Data Analysis (EDA)**: The dataset is visualized through sentiment distribution plots and word clouds.
4. **Apply TextBlob Sentiment Analysis**: TextBlob's polarity scoring is used to generate additional sentiment insights.
5. **Feature Extraction**: TF-IDF is used to convert textual data into numerical features suitable for machine learning models.
6. **Train and Evaluate the Machine Learning Model**: A Naïve Bayes classifier is trained, and its performance is evaluated using accuracy and classification reports.
7. **Visualize Model Performance**: A confusion matrix is generated to analyze misclassifications and overall effectiveness.

## Results
- The model achieves **reasonable accuracy** using Naïve Bayes for sentiment classification.
- Word clouds provide valuable insights into frequently used words in **positive** and **negative** tweets.
- The **TextBlob-based sentiment classification** provides an alternative approach to sentiment detection.

## Future Improvements
- Implement **real-time Twitter scraping** using **SNScrape**.
- Improve text preprocessing by removing **stopwords** and using **stemming/lemmatization**.
- Experiment with **different models** such as Logistic Regression, Support Vector Machines (SVM), or deep learning models like LSTMs.
- Deploy the project as a **Streamlit web app** for interactive sentiment analysis.
