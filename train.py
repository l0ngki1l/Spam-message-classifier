import warnings

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from stopwords import read_stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

warnings.filterwarnings('ignore')

stopwords_path = 'stopwords\\baidu_stopwords.txt'
stopwords = read_stopwords(stopwords_path)
data_path = 'datasets\\sms_pub.csv'
sms = pd.read_csv(data_path, encoding='utf-8')
X = np.array(sms.msg_new)
y = np.array(sms.label)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

vect = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords)
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

y_pred = nb.predict(X_test_dtm)
if __name__ == '__main__':
    print("在测试集上的混淆矩阵： ")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("在测试集上的分类结果报告： ")
    print(metrics.classification_report(y_test, y_pred))
    print("在测试集上的 f1-score: ")
    print(metrics.f1_score(y_test, y_pred))

