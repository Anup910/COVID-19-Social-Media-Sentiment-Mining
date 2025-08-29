## COVID-19-Social-Media-Sentiment-Mining
# COVID-19 Social Media Sentiment Mining | Python, scikit-learn, NLTK, PyTorch, Transformers
Analyzed 41,000 tweets from the Corona_NLP dataset to track public sentiment (Positive, Neutral, Negative) during the COVID-19 pandemic. Designed a comprehensive NLP pipeline (lowercasing, hashtag/mention/URL removal, punctuation cleanup, stopword removal, lemmatization) and conducted EDA with word clouds and trend analysis.
Benchmarked traditional models (Logistic Regression, SVM) achieving macro-F1 ~0.71, then fine-tuned a BERT transformer, raising macro-F1 to 0.80 (+9%), with class-level performance of Positive (0.83), Negative (0.79), Neutral (0.76).
The model trained in ~25 minutes over 5 epochs and achieved 80% validation accuracy on 4.1k unseen tweets. Delivered insights into population mood dynamics, enabling public-health teams and communicators to adapt messaging strategies in real time.
