# Big_Data_News_Category_Analysis
Large-scale Natural Language Processing (NLP) analysis on news articles using PySpark, HuggingFace Transformers, BERTopic, LDA, and spaCy.

## 📈 Project Overview

This project performs large-scale Natural Language Processing (NLP) analysis on news articles using PySpark, HuggingFace Transformers, BERTopic, LDA, and spaCy.
https://www.kaggle.com/datasets/rmisra/news-category-dataset 

**Main Tasks:**

* Sentiment Analysis with DistilBERT and BERT
* Topic Modeling using LDA and BERTopic
* Named Entity Recognition (NER) using spaCy
* Visualization of Sentiments, Topics, and NER

## 💡 Features Implemented

### 1. Sentiment Analysis

* Uses `distilbert-base-uncased-finetuned-sst-2-english` and `nlptown/bert-base-multilingual-uncased-sentiment`
* Compares outputs and calculates agreement/accuracy
* Visualized via bar charts and agreement charts

### 2. Text Preprocessing

* Cleaned and normalized text (lowercased, links/punctuation removed)
* Used spaCy for:

  * Tokenization
  * Stopword removal
  * Lemmatization

### 3. Topic Modeling

* **LDA** via `sklearn`
* **BERTopic** via `bertopic`
* Outputs compared via bar charts and topic overlap heatmap

### 4. NER (Named Entity Recognition)

* Used `spaCy`'s pre-trained `en_core_web_sm` model
* Top entity types extracted and visualized

## 📊 Output Files

All outputs are saved under the `outputs/` directory:

* `distilbert_sentiment_distribution.png`
* `bert_sentiment_distribution.png`
* `sentiment_agreement.png`
* `short_description_wordcloud.png`
* `lda_topic_X_wordcloud.png`
* `bertopic_topic_distribution.png`
* `topic_doc_counts_side_by_side.png`
* `topic_overlap_heatmap.png`
* `ner_entity_distribution.png`

## 🚀 Running the Project

### Requirements

```bash
pip install pyspark pandas matplotlib seaborn scikit-learn spacy wordcloud transformers bertopic
python -m spacy download en_core_web_sm
```

### Execution

Make sure Hadoop and Spark are running, then:

```bash
python data.py
```

Ensure your HDFS path matches the dataset path:

```py
hdfs://localhost:9000/project/News_Category_Dataset_v3.json
```

## 🔗 Contributors

* Namra Ahmer (23k-7301)
* Yusra Khan Baloch (23k-8046)

## ✅ Notes

* Sampling is used (`limit(500)` and `[:2000]`) to avoid memory issues
* Agreement-based accuracy is used since ground truth labels are absent

Let us know if you want to add model benchmarking, export results as CSV, or deploy this as a web interface.
