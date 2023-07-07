# Natural Language Processing with Disaster Tweets
- This repository contains the code and explanation for our solution to the Natural Language Processing with Disaster Tweets competition on Kaggle. Our model distinguishes tweets that refer to real disasters from the ones that do not.

## Overview of the Competition
- Twitter has become a significant communication channel in times of emergency. Real-time reporting through smartphones enables people to announce emergencies they're witnessing. As a result, many organizations (e.g., disaster relief organizations, news agencies) show a growing interest in monitoring Twitter programmatically.

- However, it's not always apparent whether a tweet is genuinely announcing a disaster. Therefore, in this competition, we're challenged to build a machine learning model that predicts which tweets refer to real disasters and which ones don't.

## Dataset
- The dataset used for this competition comprises 10,000 tweets that were manually classified and was created by the company Figure Eight. It was originally shared on their 'Data For Everyone' website here.

## Our Approach
- We started our journey with an SVM model that initially provided a decent accuracy of 76.69%. However, we then transitioned to a sophisticated RoBERTa model using PyTorch and K-Fold cross-validation, and that escalated our accuracy to 84.09%. This significant step improved our model's accuracy by 9.65%.

- We further pushed the boundaries by employing advanced optimization techniques such as Cython and Multiprocessing, which substantially reduced our preprocessing time by 49.36%, SVM training time by 99.91%, and RoBERTa model training time by 36.48%. These steps significantly enhanced our runtime efficiency.

- This multi-pronged strategy ensured our model's top-tier performance, securing our position in the top 3% out of 1200 teams in the Kaggle competition.

## Code
- The Jupyter notebook NLPDT.ipynb contains our complete code and approach to the problem.

## Results
- Our best model achieved an accuracy of 84.09% on the test data, which secured us a position in the top 3% of the competition.

## Conclusion
- The competition provided us with an exciting opportunity to work on a real-world NLP problem. We are pleased with the results we achieved. This project reaffirmed the power of advanced NLP models like RoBERTa and the potential of optimization techniques in improving the efficiency of machine learning models.
