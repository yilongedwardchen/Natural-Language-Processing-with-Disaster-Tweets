# Natural Language Processing with Disaster Tweets
This repository contains the code and explanation for our solution to the Natural Language Processing with Disaster Tweets competition on Kaggle. Our model distinguishes tweets that refer to real disasters from the ones that do not.

## Overview of the Competition
Twitter has become a significant communication channel in times of emergency. Real-time reporting through smartphones enables people to announce emergencies they're witnessing. As a result, many organizations (e.g., disaster relief organizations, news agencies) show a growing interest in monitoring Twitter programmatically.

However, it's not always apparent whether a tweet is genuinely announcing a disaster. Therefore, in this competition, we're challenged to build a machine learning model that predicts which tweets refer to real disasters and which ones don't.

## Dataset
The dataset used for this competition comprises 10,000 tweets that were manually classified and was created by the company Figure Eight. It was originally shared on their 'Data For Everyone' website here. The dataset is divided into training and testing sets, which are stored in train.csv and test.csv files respectively.

## Our Approach
Our exploration started with an SVM model that initially provided a decent accuracy of 76.69%. Eager to improve the results, we experimented with several state-of-the-art NLP models, such as BERT, RoBERTa, XLNet, and ELECTRA.

After comprehensive testing and analysis, we discovered that the RoBERTa-large model was most suitable for our dataset. We then applied K-Fold cross-validation on this selected model, escalating our accuracy to an impressive 84.09%. This significant step marked an accuracy improvement of 9.65%.

Next, we further pushed the boundaries by employing advanced optimization techniques such as Cython and Multiprocessing. These strategies substantially reduced our preprocessing time by 49.36%, SVM training time by 99.91%, and RoBERTa model training time by 36.48%. These steps significantly enhanced our runtime efficiency.

## Code
- The Jupyter notebook NLPDT-SVM.ipynb contains our code part using SVM approach.
- The Jupyter notebook NLPDT-T.ipynb contains our code part using functions and models in Transformers Library to approach the problem.

## Results
Our best model, the optimised RoBERTa-large model, achieved an accuracy of 84.09% on the test data, which secured us a position in the top 3% of the competition.

## Conclusion
The competition provided us with an exciting opportunity to work on a real-world NLP problem. It allowed us to compare the performance of various advanced NLP models on the same dataset, leading us to choose the RoBERTa-large model. We are pleased with the results we achieved. This project reaffirmed the power of advanced NLP models like RoBERTa and the potential of optimization techniques in improving the efficiency of machine learning models.
