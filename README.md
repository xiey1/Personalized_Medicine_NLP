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
* [Part IV: Latent Semantic Analysis](#PartIV_link)
* [Part V: Latent Semantic Analysis + `Gene` and `Variation` information](#PartV_link)
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
**CountVectorizer** from sklearn is used to convert texts to a matrix of token counts. Machine learning models **LogisticRegression**, **RandomForestClassifier** and **XGBClassifier** will then be applied to the count representation (sparse matrix) of texts.
* Total number of features in CountVectorizer: 157815
* Classification result and evaluation: 
  **XGBClassifier(eta=0.05,max_depth=6,min_child_weight=10,gamma=0,colsample_bytree=0.6)** achieves the highest accuracy score of 0.48.
<img src= 'https://github.com/xiey1/Personalized_Medicine_NLP/blob/master/images/CountVectorizer_confusion_matrix.png' width=500px>
<img src= 'https://github.com/xiey1/Personalized_Medicine_NLP/blob/master/images/CountVectorizer_barplot.png' width=500px>

Consistent with the previous EDA result, the majority of misclassified `Class2` texts fall into `Class7`.

<a id='PartIII_link'></a>
## Part III: TfidfVectorizer + Machine Learning models
**TfidfVectorizer** from sklearn is used to convert texts to a matrix of token counts. Machine learning models **LogisticRegression**, **RandomForestClassifier** and **XGBClassifier** will then be applied to the count representation (sparse matrix) of texts.
* Total number of features in TfidfVectorizer: 157815
* Classification result and evaluation: 
  **XGBClassifier(eta=0.05,max_depth=6,min_child_weight=5,gamma=0.4,colsample_bytree=0.2)** achieves the highest accuracy score of 0.50.
<img src= 'https://github.com/xiey1/Personalized_Medicine_NLP/blob/master/images/TfidfVectorizer_confusion_matrix.png' width=500px>
<img src= 'https://github.com/xiey1/Personalized_Medicine_NLP/blob/master/images/TfidfVectorizer_barplot.png' width=500px>

1. The overall training performance is comparable between **CountVectorizer** and **TfidfVectorizer**. 
2. Again, consistent with the previous EDA result, the majority of misclassified `Class2` texts fall into `Class7`.

<a id='PartIV_link'></a>
## Part IV: Latent Semantic Analysis
**Truncated singular value decomposition (SVD)** is applied to count matrix obtained from **CountVectorizer** or **TfidfVectorizer** to perform linear dimensionality reduction. Machine learning models **LogisticRegression**, **RandomForestClassifier** and **XGBClassifier** will then be applied to the matrix with reduced dimension.

* Classification result and evaluation: 
  **TruncatedSVD(n_components=50) + RandomForestClassifier(max_depth=10,min_samples_leaf=15)** achieves the highest accuracy score of 0.42.
<img src= 'https://github.com/xiey1/Personalized_Medicine_NLP/blob/master/images/LSA_confusion_matrix.png' width=500px>
<img src= 'https://github.com/xiey1/Personalized_Medicine_NLP/blob/master/images/LSA_barplot.png' width=500px>

1. The overall training performance for LSA method is not better than BOW using CountVectorizer and TfidfVectorizer, with overall validation accuracy around 0.40. 
2. Larger n_components value in TruncatedSVD leads to increased bias and overfitting. Generally n_components=25 or 50 achieves better performance.

<a id='PartV_link'></a>
## Part V: Latent Semantic Analysis + `Gene` and `Variation` information
One-hot-encoding and Truncated singular value decomposition (SVD) is applied to transform Gene and Variation data. The obtained matrix is concatenated with the output of LSA. Machine learning models **LogisticRegression**, **RandomForestClassifier** and **XGBClassifier** will then be applied to the combined matrix containing information about both text and gene variation.

* Classification result and evaluation:
Here I use n_components=50 to perform LSA and use n_components=25 to perform dimension reduction for `Gene` and `Variation` information. The combined matrix containing information about both text and gene variation has a dimension of 100.
  **TruncatedSVD(n_components=50) + StandardScaler + LogisticRegression(C=0.0002)** achieves the highest accuracy score of 0.43.
<img src= 'https://github.com/xiey1/Personalized_Medicine_NLP/blob/master/images/LSA2_confusion_matrix.png' width=500px>
<img src= 'https://github.com/xiey1/Personalized_Medicine_NLP/blob/master/images/LSA2_barplot.png' width=500px>

The overall training performance for LSA method with gene variation information is not better than LSA or BOW method, with overall validation accuracy around 0.40.

<a id='PartVI_link'></a>
## Part VI: Word2Vec + Machine Learning models
**Word2Vec** is applied to train an embedding matrix in which each word is represented by a numeric vector with a fixed dimension containing its semantic meaning. The numeric vectors for words in each document are averaged to obtain a vecotr to represent each document. The obtained matrix can be concatenated with the transformed matrix containing information about `Gene` and `Variation`. Machine learning models **LogisticRegression**, **RandomForestClassifier** and **XGBClassifier** will then be applied to the matrix either only containing text information or both text and gene variation data.

* Word2Vec result:
Here I choose to use embedding size as 100. After conversion, each word will be represented by a numeric vector with dimension as 100. **t-SNE** (t-distributed Stochastic Neighbor Embedding) plot can then be applied to visualize the spatial distribution of numeric representations of the words with semantic meanings similar to `benign` and `malignant` in t-SNE plot.
<img src= 'https://github.com/xiey1/Personalized_Medicine_NLP/blob/master/images/Word2Vec_tsne.png' width=500px>
The Word2Vec embedding matrix can transform words to numeric vectors with specific semantic meanings in terms of tumor malignancy.

We can also visualize the spatial distribution of text from each class using t-SNE plot.
<img src= 'https://github.com/xiey1/Personalized_Medicine_NLP/blob/master/images/Word2Vec_tsne2.png' width=500px>
We cannot observe distinct clusters using t-SNE plot, which suggests that machine learning models are needed to further classify each document.

* Classification result and evaluation:
Here I use n_components=25 to perform dimension reduction for `Gene` and `Variation` information. The combined matrix containing information about both text and gene variation has a dimension of 150 (100+25+25).
  **Word2Vec + StandardScaler + LogisticRegression(C=0.001)** achieves the highest accuracy score of 0.51.
<img src= 'https://github.com/xiey1/Personalized_Medicine_NLP/blob/master/images/Word2Vec_confusion_matrix.png' width=500px>
<img src= 'https://github.com/xiey1/Personalized_Medicine_NLP/blob/master/images/Word2Vec_barplot.png' width=500px>

Word2Vec model with gene variation information performs slightly better, with overall validation accuracy around 0.51 for LogisticRegression(C=0.001)
