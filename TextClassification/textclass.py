from sklearn import model_selection,preprocessing,linear_model,naive_bayes,metrics,svm
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn import decomposition,ensemble

import pandas,xgboost,numpy,textblob,string
from keras.preprocessing import text,sequence
from keras import layers,models,optimizers

#一、准备数据集
#加载数据集
data = open('data/corpus').read()
labels, texts = [],[]
for i,line in enumerate(data.split("\n"))
content = line.split()
labels.append(content[0])
texts.append(content[1])

#创建一个dataframe，列名为text和label
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

#将数据集分为训练集和验证集
train_x,valid_x,train_y,valid_y = model_selection.train_test_split(trainDF['text'],trainDF['label'])#默认训练比例为0.75

#label编码为目标变量
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

#二、特征工程
#创建一个向量计数器对象
count_vect = CountVectorizer(analyzer = 'word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])

#使用向量计数器对象转换训练集和验证集
xtrain_count = count_vect.transform(train_x)
xvalid_count = count_vect.transform(valid_x)

#词语级tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word',token_pattern=r'\w{1,}',max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)

#n-gram级别的tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer=)
