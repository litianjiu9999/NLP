import jieba
text = "我今天吃了六个豆沙包，早上、中午、晚上各两个。"
# jieba.cut得到的是generator形式的结果
seg = jieba.cut(text)  
print(' '.join(seg)) 
##test


import jieba.posseg as posseg
text = "一天不敲代码我就浑身难受"
# 形如pair('word, 'pos')的结果
seg = posseg.cut(text)  
print([se for se in seg]) 
