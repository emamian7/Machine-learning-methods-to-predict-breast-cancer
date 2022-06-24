# Machine-learning-methods-to-predict-breast-cancer

Abstract: Nowadays, breast cancer is dangerous for woman however in term of health care, they
collect data regard how malignant or benign from breast cancer versus 10 real-valued features 
which are computed for each cell nucleus. For this project we will preparing the data, selecting the 
features from distribution, choosing and applying the machine learning, comparing and improving 
the best models.

Conclusion:We use data set from Wisconsin (Diagnostic) to determine whether the cancer is benign 
or malignant by apply SGD, SVC, NuSVC, LinearSVC, KNN with k=5, Naïve Bayes, Random 
Forest, Extra Trees and Decision Trees on all mean data set and selected feature from normal 
distribution which only SGD, SVC, NuSVC and LinearSVC has increase accuracy significant 
except Extra Trees and Decision Trees has slightly accuracy increase around 1%. Naïve Bayes 
has no any improvement but in contrast Random Forest has decrease accuracy.
Moreover, after we apply grid search we can see that only Extra Trees has improve 
accuracy but Decision Trees has reduce accuracy around 0.87%. However, Naïve Bayes and 
Random Forest has no improvement.
Thus, Naïve Bayes with grid search given the highest accuracy 94.76% for classify 
cancer is benign or malignant from Wisconsin (Diagnostic) Data set.

Data Set
Predict whether the cancer is benign or malignant from Wisconsin (Diagnostic) Data Set. Features 
are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass including
