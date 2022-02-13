import jieba
text = "我今天吃了六个豆沙包，早上、中午、晚上各两个。"
# jieba.cut得到的是generator形式的结果
seg = jieba.cut(text)  
print(' '.join(seg)) 
