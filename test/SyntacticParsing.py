from nltk import word_tokenize, pos_tag
text = "I am going to read this book in the flight"
tokens = word_tokenize(text)
print(tokens)
print(pos_tag(tokens))