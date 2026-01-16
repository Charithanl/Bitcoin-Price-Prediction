import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
text = "python is easy to learn. it is widely used in data science."  
tokens = sent_tokenize(text)
print("token:")
print(tokens)