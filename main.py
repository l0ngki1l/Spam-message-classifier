import pandas as pd
from stopwords import read_stopwords
data_path = 'datasets\\sms_pub.csv'
sms = pd.read_csv(data_path, encoding='utf-8')
sms.head()

stopwords_path = 'stopwords\\baidu_stopwords.txt'
stopwords = read_stopwords(stopwords_path)

