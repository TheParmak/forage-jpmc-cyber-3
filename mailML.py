import re
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import sklearn.linear_model as linear_model


def read_spam():
    category = 'spam'
    directory = './enron1/spam'
    return read_category(category, directory)

def read_ham():
    category = 'ham'
    directory = './enron1/ham'
    return read_category(category, directory)

def read_category(category, directory):
    emails = []
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(directory, filename), 'r') as fp:
            try:
                content = fp.read()
                emails.append({'name': filename, 'content': content, 'category': category})
            except:
                pass
                # print(f'skipped {filename}')
    return emails

ham = read_ham()
spam = read_spam()

df = pd.DataFrame.from_records(ham)
df = df.append(pd.DataFrame.from_records(spam))



def preprocess(e):
    e = re.sub(r'[^a-zA-Z]', ' ', e).lower()
    return e


def scivec():
    data = df['content']
    labels = df['category']

    vectorizer = CountVectorizer(preprocessor=preprocess)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    X_test_vectorized = vectorizer.fit_transform(X_train)

    logreg = linear_model.LogisticRegression(max_iter=400)
    logreg.fit(X_test_vectorized, y_train)
    
    X_test_vectorized = vectorizer.transform(X_test)
    prediction = logreg.predict(X_test_vectorized)
    
    print(accuracy_score(y_test, prediction))
    print(confusion_matrix(y_test, prediction, normalize='true'))
    # scikitplot.metrics.plot_confusion_matrix(y_test, prediction, normalize=True)
    print(classification_report(y_test, prediction))

    feature_names = vectorizer.get_feature_names_out()
    # print(feature_names)

    coef = logreg.coef_[0]
    print(coef)
    importances = logreg.coef_[0]

    feature_importances = zip(importances, feature_names)
    fsort = sorted(feature_importances, reverse=True)

    print("\nTop 10 Positive Feature Importances (SPAM):")
    for importance, feature_name in fsort[:10]:
        print(f"SPAM: Feature: {feature_name}, Importance: {importance}")


    print("\nTop 10 Negative Feature Importances (HAM):")
    for importance, feature_name in reversed(fsort[-10:]):
        print(f"HAM: Feature: {feature_name}, Importance: {importance}")

        

scivec()

