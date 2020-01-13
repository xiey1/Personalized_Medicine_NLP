# Personalized_Medicine_NLP
Classification of genetic variations based on clinical evidence (text)
<br>Data source: https://www.kaggle.com/c/msk-redefining-cancer-treatment/data

# Project Aim:
### Create an algorithm to classify genetic mutations based on clinical evidence (text description) of genetic mutations
Multiclass classification of clinical evidence (text) into 9 classes. However, the criterion by which each class is defined is not provided. In addition to text, `Gene` and `Variation` information of each mutation is also provided, which could be used in combination with text description to classify genetic mutations. 

# Approach:
### Text transformation models:
  CountVectorizer
  TfidfVectorizer
  Truncated singular value decomposition (SVD)
  Word2Vec (self-trained and pre-trained
  Doc2Vec
### Machine learning models:
  LogisticRegression
  RandomForestClassifier
  XGBClassifier
### Deep learning models
  Recurrent neural network (RNN)
  Long short-term memory (LSTM)
  Gated Recurrent Units (GRU)
  Conv1d

<br>This project has 9 sections with code and detailed explanation in 9 jupyter notebooks.

* [Part I: Exploratory data analysis](#PartI_link)
* [Part II: CountVectorizer + Machine Learning models](#PartII_link)
* [Part III: TfidfVectorizer + Machine Learning models](#PartIII_link)
* [Part IV: Latent Semantic Analysis (LSA)](#PartIV_link)
* [Part V: Latent Semantic Analysis (LSA) + `Gene` and `Variation` information](#PartV_link)
* [Part VI: Word2Vec + Machine Learning models](#PartVI_link)
* [Part VII: Doc2Vec + Machine Learning models](#PartVII_link)
* [Part VIII: Deep Learning models with pre-trained Word2Vec](#PartVIII_link)
* [Part IX: Deep Learning models with self-trained Word2Vec](#PartIX_link)

<a id='PartI_link'></a>
## Part I: Exploratory data analysis
### Short overview of the dataset:
  Number of documents in training dataset: 3321
  Number of documents in testing dataset: 5668
  
  After removing documents with NA in `Text` field:
  Number of documents in training dataset: 3316
  Number of documents in testing dataset: 5667

* The training dataset is imbalanced and the number of images in each dataset and in each class is shown as below.
<img src= 'https://github.com/xiey1/Personalized_Medicine_NLP/blob/master/images/class_description.png' width=600px>

There are 9 classes of genetic mutations in this project and the distribution of data is imbalanced. `Class7` and `Class4` account for almost 50% of the entire dataset, while `Class3`, `Class8` and `Class9` have very few descriptions. The imbalance in data distribution needs to be considered when we split datasets as well as train the classification model.

* Distribution of the number of sentences and number of words in texts for each class.
<img src= 'https://github.com/xiey1/Personalized_Medicine_NLP/blob/master/images/word_sentence_length_distplot.png' width=600px>
<img src= 'https://github.com/xiey1/Personalized_Medicine_NLP/blob/master/images/word_sentence_length_violinplot.png' width=600px>

`Class7-9` have similar distribution of the number of sentences and the number of words in each class. However, `Class8` and `Class9` have large variation in word/sentence length distribution, which may be due to the small sample size

* The most frequently mutated genes for each class.
<img src= 'https://github.com/xiey1/Personalized_Medicine_NLP/blob/master/images/gene_perclass.png' width=600px>

`Class2` and `Class7` have similar most frequently mutated genes.

* Keywords for each class.
<img src= 'https://github.com/xiey1/Personalized_Medicine_NLP/blob/master/images/keywords_perclass.png' width=600px>
<img src= 'https://github.com/xiey1/Personalized_Medicine_NLP/blob/master/images/WordCloud_perclass.png' width=600px>

Again, `Class2` and `Class7` have similar keywords with enrichment of `EGFR` mutations and `patient` as keyword. Both `Class2` and `Class7` are also highly enriched with `KIT` and `BRAF` mutations based on the mutant gene profiling. `Class5` and `Class6` are characterized by `BRCA1` mutations. `Class8` and `Class9` are both very small datasets and enriched with `IDH1` and `IDH2` mutations.

It is not clear the criterion that mutations are classified in this project. Based on the exploratory data analysis, genetic profiling plays an important role in classification.

<a id='PartII_link'></a>
## Part II: CountVectorizer + Machine Learning models
CountVectorizer from sklearn is used to convert texts to a matrix of token counts. Machine learning models LogisticRegression, RandomForestClassifier and XGBClassifier will then be applied to the count representation (sparse matrix) of texts.
* Total number of features in CountVectorizer: 157815
* Coclusion: 
  **XGBClassifier(eta=0.05,max_depth=6,min_child_weight=10,gamma=0,colsample_bytree=0.6)** achieves the highest accuracy score of 0.48.
<img src= 'https://github.com/xiey1/Personalized_Medicine_NLP/blob/master/images/CountVectorizer_confusion_matrix.png' width=500px>
<img src= 'https://github.com/xiey1/Personalized_Medicine_NLP/blob/master/images/CountVectorizer_barplot.png' width=500px>
