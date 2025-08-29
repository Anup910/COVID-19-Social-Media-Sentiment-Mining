## COVID-19-Social-Media-Sentiment-Mining
# COVID-19 Social Media Sentiment Mining | Python, Pandas , scikit-learn, NLTK, PyTorch, Transformers

Analyzed 41,000 tweets from the Corona_NLP dataset to track public sentiment (Positive, Neutral, Negative) during the COVID-19 pandemic. Designed a comprehensive NLP pipeline (lowercasing, hashtag/mention/URL removal, punctuation cleanup, stopword removal, lemmatization) and conducted EDA with word clouds and trend analysis.

Benchmarked traditional models (Logistic Regression, SVM) achieving macro-F1 ~0.71, then fine-tuned a BERT transformer, raising macro-F1 to 0.80 (+9%), with class-level performance of Positive (0.83), Negative (0.79), Neutral (0.76).

The model trained in ~25 minutes over 5 epochs and achieved 80% validation accuracy on 4.1k unseen tweets. Delivered insights into population mood dynamics, enabling public-health teams and communicators to adapt messaging strategies in real time.

## Covid-19 Sentiment Analysis 🦠📊
📌 Project Overview

This project applies Natural Language Processing (NLP) and Machine Learning techniques to analyze public sentiment related to Covid-19. The goal is to uncover how people reacted on social media, news, and open datasets during the pandemic, providing insights into fear, positivity, and public opinion trends.

## 🛠️ Tools & Technologies

Python (Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Seaborn, WordCloud)

Jupyter Notebook for interactive analysis

Data Visualization for sentiment distribution and word frequency analysis

## 📂 Dataset

Covid-19 related text data (tweets, comments, or open repositories).

Preprocessed by cleaning, tokenization, stopword removal, and lemmatization.

## 📊 Key Insights

Conducted Exploratory Data Analysis (EDA) on Covid-19 sentiment data.

Built word clouds and frequency plots to highlight common terms.

Classified text into positive, neutral, and negative sentiments.

Identified temporal sentiment patterns and key shifts during pandemic peaks.

## 🚀 Impact

This analysis helps:

Researchers & policymakers understand public perception.

Healthcare organizations monitor fear and misinformation trends.

Businesses adapt communication strategies during crises.

## 📈 Sample Visuals

(Word Clouds, Sentiment Distribution, and Trend Graphs are included in the notebook and render on GitHub.)
