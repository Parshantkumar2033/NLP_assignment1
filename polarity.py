import os
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')

def create_dataset(positive, negative):
    pos_data = pd.read_csv(positive, encoding = 'Windows-1252')
    neg_data = pd.read_csv(negative, encoding = 'Windows-1252')
    pos_data['sentiment'] = np.ones(pos_data.shape[0], dtype = np.int8)
    neg_data['sentiment'] = np.zeros(neg_data.shape[0], dtype = np.int8)
    train = shuffle(pd.concat([pos_data.iloc[:4000, :], neg_data.iloc[:4000, :]], axis = 0), random_state=71).reset_index(drop = True)
    x_train = train.iloc[:, 0]
    y_train = train.iloc[:, 1]
    val = shuffle(pd.concat([pos_data.iloc[4000:4500, :], neg_data.iloc[4000:4500, :]], axis = 0), random_state=71).reset_index(drop = True)
    x_val = val.iloc[:, 0]
    y_val = val.iloc[:, 1]
    test = shuffle(pd.concat([pos_data.iloc[4500:, :], neg_data.iloc[4500:, :]], axis = 0), random_state=71).reset_index(drop = True)
    x_test = test.iloc[:, 0]
    y_test = test.iloc[:, 1]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# Training
positive = 'positive.csv'
negative = 'negative.csv'
(X_train, y_train), (X_val, y_val), (X_test, y_test) = create_dataset(positive, negative)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# Train classifiers
model_svm = LinearSVC()
model_nb = MultinomialNB()

model_svm.fit(X_train_tfidf, y_train)
model_nb.fit(X_train_tfidf, y_train)

# Evaluate on the validation set
val_pred_svm = model_svm.predict(X_val_tfidf)
val_pred_nb = model_nb.predict(X_val_tfidf)

# Confusion matrices
confusion_svm = classification_report(y_val, val_pred_svm)
confusion_nb = classification_report(y_val, val_pred_nb)
conf_matirx_svm = confusion_matrix(y_val, val_pred_svm)
conf_matrix_nb = confusion_matrix(y_val, val_pred_nb)

# results
print("\n\n\033[1mValidation\033[0m\n\n")
print("\033[1mValidation set SVM\033[0m\n", confusion_svm)
print("\033[1mConfusion Matrix\033[0m \n", conf_matirx_svm)

print("\033[1mValidation set NB\033[0m\n", confusion_nb)
print("\033[1mConfusion Matrix\033[0m \n", conf_matrix_nb)


# Evaluate on the test set
test_pred_svm = model_svm.predict(X_test_tfidf)
test_pred_nb = model_nb.predict(X_test_tfidf)

# Confusion matrices for the test set
confusion_svm_test = classification_report(y_test, test_pred_svm)
confusion_nb_test = classification_report(y_test, test_pred_nb)

conf_matirx_svm_test = confusion_matrix(y_test, test_pred_svm)
conf_matrix_nb_test = confusion_matrix(y_test, test_pred_nb)

# results
print("\n\n\033[1mTest\033[0m\n\n")
print("\033[1mConfusion matrix for SVM on test set:\033[0m\n", confusion_svm_test)
print("\033[1mConfusion Matrix\033[0m \n", conf_matirx_svm_test)

print("\033[1mConfusion matrix for Naive Bayes on test set:\033[0m\n", confusion_nb_test)
print("\033[1mConfusion Matrix\033[0m \n", conf_matrix_nb_test)

