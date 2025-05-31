# Converted from Untitled2.ipynb with Sentiment (Transformer), Topic Modeling (LDA), NER (spaCy), and Visualizations

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import spacy
from collections import Counter
from transformers import pipeline
import re
from bertopic import BERTopic


# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # remove links
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation and numbers
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra whitespace

    # Tokenize, remove stopwords, and lemmatize
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.lemma_ != '-PRON-']
    return " ".join(tokens)

# Start Spark session
spark = SparkSession.builder \
    .appName("Sentiment, Topic Modeling, and NER") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Load JSON file from HDFS
print("\n✅ Loading data...")
df = spark.read.json("hdfs://localhost:9000/project/News_Category_Dataset_v3.json")

# Select necessary columns and clean data
df = df.select("category", "headline", "short_description", "link", "date")
df = df.dropna(subset=["short_description"])
print("\n✅ Total rows:", df.count())

# Sample only part of data to avoid memory errors
df_sample = df.limit(500)
df_pd = df_sample.select("short_description", "date").toPandas()

# Apply text preprocessing
df_pd['short_description'] = df_pd['short_description'].apply(preprocess_text)

# Load transformer sentiment pipelines
print("🤖 Loading transformer sentiment models...")
sentiment_model_distilbert = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
sentiment_model_bert = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Apply DistilBERT sentiment to sample data
def get_distilbert_sentiment(text):
    try:
        result = sentiment_model_distilbert(text[:512])[0]
        return result['label'].lower()
    except:
        return "neutral"

df_pd['distilbert_sentiment'] = df_pd['short_description'].apply(get_distilbert_sentiment)

# Apply BERT sentiment to sample data
def get_bert_sentiment(text):
    try:
        result = sentiment_model_bert(text[:512])[0]
        label = result['label'].lower()
        if '1' in label or '2' in label:
            return 'negative'
        elif '3' in label:
            return 'neutral'
        else:
            return 'positive'
    except:
        return "neutral"

df_pd['bert_sentiment'] = df_pd['short_description'].apply(get_bert_sentiment)

# Convert date to datetime for plotting
df_pd['date'] = pd.to_datetime(df_pd['date'], errors='coerce')
df_pd = df_pd.dropna(subset=['date'])

# Sentiment distribution charts
print("📊 Plotting DistilBERT sentiment distribution")
distilbert_counts = df_pd['distilbert_sentiment'].value_counts()
distilbert_counts.plot(kind='bar', color='salmon', figsize=(8, 4), title='DistilBERT Sentiment Distribution')
plt.ylabel("Article Count")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("outputs/distilbert_sentiment_distribution.png")
plt.show()

print("📊 Plotting BERT sentiment distribution")
bert_counts = df_pd['bert_sentiment'].value_counts()
bert_counts.plot(kind='bar', color='lightblue', figsize=(8, 4), title='BERT Sentiment Distribution')
plt.ylabel("Article Count")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("outputs/bert_sentiment_distribution.png")
plt.show()

# Compare agreement between models
print("📊 Comparing agreement between DistilBERT and BERT")
df_pd['agreement'] = df_pd['distilbert_sentiment'] == df_pd['bert_sentiment']
agreement_counts = df_pd['agreement'].value_counts()
agreement_counts.index = ['Disagree', 'Agree']
agreement_counts.plot(kind='bar', color=['orangered', 'mediumseagreen'], title='Sentiment Agreement: DistilBERT vs BERT', figsize=(8, 4))
plt.ylabel("Article Count")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("outputs/sentiment_agreement.png")
plt.show()

# Calculate sentiment agreement accuracy
matching_count = (df_pd['distilbert_sentiment'] == df_pd['bert_sentiment']).sum()
total_count = len(df_pd)
accuracy = matching_count / total_count
print(f"🎯 Sentiment Agreement Accuracy between DistilBERT and BERT: {accuracy:.2%}")

# Accuracy per model assuming human-like consensus between the two
model_agreement = df_pd[df_pd['distilbert_sentiment'] == df_pd['bert_sentiment']]
distilbert_accuracy = (model_agreement['distilbert_sentiment'] == model_agreement['bert_sentiment']).mean()
bert_accuracy = (model_agreement['bert_sentiment'] == model_agreement['distilbert_sentiment']).mean()

print(f"📈 Estimated DistilBERT Accuracy (vs. BERT): {distilbert_accuracy:.2%}")
print(f"📈 Estimated BERT Accuracy (vs. DistilBERT): {bert_accuracy:.2%}")
matching_count = (df_pd['distilbert_sentiment'] == df_pd['bert_sentiment']).sum()
total_count = len(df_pd)
accuracy = matching_count / total_count
print(f"🎯 Sentiment Agreement Accuracy between DistilBERT and BERT: {accuracy:.2%}")

# WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(df_pd['short_description'].astype(str)))
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Most Frequent Terms in Short Descriptions")
plt.tight_layout()
plt.savefig("outputs/short_description_wordcloud.png")
plt.show()

# ------------------ Topic Modeling ------------------
print("🔍 Performing Topic Modeling with LDA and BERTopic...")

# Take only 2000 preprocessed documents
documents = df_pd["short_description"].dropna().tolist()[:2000]

# LDA
vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=5, max_features=5000)
doc_term_matrix = vectorizer.fit_transform(documents)
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(doc_term_matrix)

print("📌 LDA Topics:")
def display_topics(model, feature_names, no_top_words):
    for idx, topic in enumerate(model.components_):
        words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        print(f"Topic {idx+1}: ", " ".join(words))
        wordcloud = WordCloud(width=800, height=400).generate(" ".join(words))
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"LDA Topic {idx+1} WordCloud")
        plt.tight_layout()
        plt.savefig(f"outputs/lda_topic_{idx+1}_wordcloud.png")

feature_names = vectorizer.get_feature_names_out()
display_topics(lda, feature_names, 10)

# BERTopic
print("📌 Running BERTopic...")

bertopic_model = BERTopic(verbose=False)
bertopic_topics, _ = bertopic_model.fit_transform(documents)

# Save and plot BERTopic results
topic_freq = bertopic_model.get_topic_info()
topic_freq[topic_freq.Topic != -1].head(10).plot(kind='bar', x='Topic', y='Count', title='BERTopic Top Topics', color='purple', figsize=(8,4))
plt.ylabel("Document Count")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("outputs/bertopic_topic_distribution.png")

print("✅ Topic word clouds and BERTopic graph saved to outputs")

# Assign dominant LDA topic to each document
lda_topic_distributions = lda.transform(doc_term_matrix)
dominant_lda_topics = lda_topic_distributions.argmax(axis=1)

# Assign topics from BERTopic
bert_topic_assignments = bertopic_topics

# Count number of documents assigned to each topic for LDA
lda_topic_counts = pd.Series(dominant_lda_topics).value_counts().sort_index()

# Count number of documents assigned to each topic for BERTopic
bert_topic_counts = pd.Series(bert_topic_assignments).value_counts().sort_index()

# If BERTopic has an outlier topic (-1), we can label it as "Outliers" (-1), we can label it as "Outliers"
if -1 in bert_topic_counts.index:
    print(f"\nNumber of outlier documents (no topic assigned by BERTopic): {bert_topic_counts[-1]}")
    bert_topic_counts = bert_topic_counts.drop(index=-1)

# Plot side-by-side bar charts for LDA vs BERTopic topic frequencies
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# LDA topics bar chart
axes[0].bar(lda_topic_counts.index.astype(str), lda_topic_counts.values, color="skyblue")
axes[0].set_title("Documents per LDA Topic")
axes[0].set_xlabel("LDA Topic")
axes[0].set_ylabel("Number of Documents")
plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")

# BERTopic topics bar chart
axes[1].bar(bert_topic_counts.index.astype(str), bert_topic_counts.values, color="salmon")
axes[1].set_title("Documents per BERTopic Topic")
axes[1].set_xlabel("BERTopic Topic")
axes[1].set_ylabel("Number of Documents")
plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig("outputs/topic_doc_counts_side_by_side.png", dpi=300)
plt.show()

# ------------------ Named Entity Recognition (NER) ------------------
print("\n🧠 Performing Named Entity Recognition with spaCy...")
nlp = spacy.load("en_core_web_sm")
all_text = " ".join(documents[:1000])
ner_doc = nlp(all_text)

entities = [ent.label_ for ent in ner_doc.ents]
entity_counts = Counter(entities)

plt.figure(figsize=(10, 5))
plt.bar(entity_counts.keys(), entity_counts.values(), color='cornflowerblue', edgecolor='black')
plt.title("Named Entity Recognition (NER) - Top Entities", fontsize=14, fontweight='bold')
plt.xlabel("Entity Type", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/ner_entity_distribution.png")
print("\n✅ NER entity distribution saved to outputs/ner_entity_distribution.png")

spark.stop()
