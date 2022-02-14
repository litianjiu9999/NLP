from nltk import word_tokenize
from nltk import TextCollection

sents=['i like jike','i want to eat apple','i like lady gaga']
# 首先进行分词
sents=[word_tokenize(sent) for sent in sents]

# 构建语料库
corpus=TextCollection(sents)

# 计算IDF
idf=corpus.idf('like')
print(idf)#0.4054651081081644