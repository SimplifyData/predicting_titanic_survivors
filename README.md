# predicting_titanic_survivors
Predicting survivors on the Titanic ship using data science techniques

This mini data science project was written using Python 2.
The libraries used were as follows:

1) Pandas 
2) Numpy
3) Sk-Learn (Logistics Regression & Support Vector Machines)

The data was first explored using Pandas Data frames, and cofounding metrics were defined using statistical methods.
A few parameters were analyzed using Naive Baye.
The data was then imputed and cleaned, and all the null values got replaced with the parameter's mean or mode value.
Co-founding coefficients were finalized. If they values were categorical, they were replaced with binary values.
The training data was then computed through Logistic Regression and SVM library since the not all coeffienects were continuous.
The Survival rate of the training set resulted in about 80% in prediction.
The test data was then passed through to compute the predicted survivors and saved on a CSV file.
