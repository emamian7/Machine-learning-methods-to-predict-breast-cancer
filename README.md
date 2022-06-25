# Machine-learning-methods-to-predict-cancer

Introduction:
In this experiment, different machine learning method such as Stochastic Gradient 
Descent, KNN, Support Vector Machines, Naive Bayes and Forest and tree methods are applied. Mean data from data set is used to apply with machine learning to analyze on accuracy and cross validation for each method.For 
improvement, the best performance of the classifier by Grid search and by sampling different hyper-parameter combinations was achieved. 
After that only some feature which depend on normal distribution was extracted and the same method was used to see the results change. 

Data Set
To predict whether the cancer is benign or malignant from Wisconsin (Diagnostic) Data Set. Features 
are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.


Conclusion
Naïve Bayes with grid search given the highest accuracy 94.76% for classification. 
This result has reached by implementing NuSVC, LinearSVC, KNN with k=5, Naïve Bayes SGD, SVC, Random 
Forest, Extra Trees and Decision Trees  on all mean data set and selected feature from normal 
distribution which only  NuSVC ,LinearSVC, SGD and SVC has increase accuracy significant 
except Extra Trees and Decision Trees has slightly accuracy increase around 1%. Naïve Bayes 
has no any improvement but in contrast Random Forest has decrease accuracy.
Moreover, after using grid search only Extra Trees has improve 
accuracy but Decision Trees has reduce accuracy around 0.87%. However, Naïve Bayes and 
Random Forest has no improvement. 


