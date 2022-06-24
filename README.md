# Machine-learning-methods-to-predict-breast-cancer

Introduction:
In this experiment I applied different machine learning method such as Stochastic Gradient 
Descent, Support Vector Machines, KNN, Naive Bayes and Forest and tree methods and also for 
improvement, I can search for the best performance of the classifier sampling different hyper-parameter combinations by exhaustive grid search. First of all, I used all mean data from data 
set to apply with machine learning to analyze on accuracy and cross validation for each method. 
After that I extracted only some feature which depend on normal distribution look like and 
repeated the same method to see the result how they are different. In the last section, I applied 
grid search method to improve machine learning model.

Data Set
To predict whether the cancer is benign or malignant from Wisconsin (Diagnostic) Data Set. Features 
are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.


Conclusion
I applied SGD, SVC, NuSVC, LinearSVC, KNN with k=5, Naïve Bayes, Random 
Forest, Extra Trees and Decision Trees on all mean data set and selected feature from normal 
distribution which only SGD, SVC, NuSVC and LinearSVC has increase accuracy significant 
except Extra Trees and Decision Trees has slightly accuracy increase around 1%. Naïve Bayes 
has no any improvement but in contrast Random Forest has decrease accuracy.
Moreover, after I applied grid search I can see that only Extra Trees has improve 
accuracy but Decision Trees has reduce accuracy around 0.87%. However, Naïve Bayes and 
Random Forest has no improvement.
Thus, Naïve Bayes with grid search given the highest accuracy 94.76% for classify 
cancer is benign or malignant from Wisconsin (Diagnostic) Data set.


