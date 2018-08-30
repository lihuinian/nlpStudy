from gensim.models import Word2Vec
sentences = [['data', 'science'], ['vidhya', 'science', 'data', 'analytics'],['machine', 'learning'], ['deep', 'learning']]
model = Word2Vec(sentences, min_count=1)
print(model.similarity('data','science'))