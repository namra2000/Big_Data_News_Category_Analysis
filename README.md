# 📘 Big Data NLP Project

## 📌 Project Title

**Analyzing Public Sentiment and Emerging Topics from Large-Scale News Articles using NLP and Big Data Tools**

## 👥 Group Members

* Namra (23k-7301)
* Yusra Khan Baloch (23k-8046)

## 🧠 Objectives

1. Perform sentiment analysis using transformer-based models (DistilBERT and BERT).
2. Conduct topic modeling using LDA and BERTopic.
3. Extract entities using spaCy for Named Entity Recognition (NER).
4. Visualize trends with word clouds, bar charts, and heatmaps.
5. Compare outputs from different models for validation and accuracy.

## 🛠️ Tools & Libraries Used

* Python 3.10.11
* PySpark 3.5.4
* pandas 2.2.1
* matplotlib 3.8.4
* seaborn 0.12.2
* scikit-learn 1.4.2
* wordcloud 1.9.3
* spaCy 3.7.4
* en\_core\_web\_sm (spaCy model)
* transformers 4.40.0
* torch 2.3.0
* bertopic 0.16.0

## 📂 Data Source

* [News Category Dataset (Kaggle)](https://www.kaggle.com/rmisra/news-category-dataset)
* Fields: `headline`, `short_description`, `category`, `link`, `date`

## 🚀 Project Workflow

1. Start Spark session and load JSON data from HDFS.
2. Preprocess text: lowercase, punctuation removal, stopword removal, lemmatization.
3. Apply transformer models to perform sentiment classification.
4. Plot sentiment distribution and agreement metrics.
5. Use CountVectorizer + LDA for basic topic modeling.
6. Apply BERTopic for advanced topic extraction.
7. Visualize LDA vs BERTopic comparison and topic overlap heatmap.
8. Run Named Entity Recognition (NER) on preprocessed content.
9. Save all outputs and graphs in `outputs/` folder.

## 📊 Output Visuals

* distilbert\_sentiment\_distribution.png
* bert\_sentiment\_distribution.png
* sentiment\_agreement.png
* short\_description\_wordcloud.png
* lda\_topic\_X\_wordcloud.png (X = topic number)
* bertopic\_topic\_distribution.png
* topic\_doc\_counts\_side\_by\_side.png
* topic\_overlap\_heatmap.png
* ner\_entity\_distribution.png

## 🏁 Running the Code

Make sure the following are set:

* HDFS path is active and JSON file is accessible
* All libraries installed as per versions listed
* Spark installed and added to PATH

```bash
# Sample run command
python data.py
```

## 🔍 Notes

* Only 500–2000 records are processed to avoid memory issues.
* Agreement between sentiment models used as proxy for accuracy.

## 📧 Contact

For any queries, reach out to Namra or Yusra via university email.
