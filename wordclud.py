# -*- coding: utf-8 -*-
"""
Date：Wed Jul 11 09:28:23 2018

@Author: ming
"""

import pandas as pd
import jieba
import re
from scipy.misc import imread  # 这是一个处理图像的函数
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer





# 创建停用词表
def stop_word_list(filepath):
    stop_word = [line.strip() for line in open(filepath, 'r', encoding='UTF-8').readlines()]
    return stop_word



# 去除停用词

def Word_cut(data):
    file = 'D:\\py\\emotion\\data\\stop_words.txt'
    new_seg = []
    stop_words = stop_word_list(file)
    for i in range(len(data)):
        line = data[i]
        # 中文的编码范围是：\u4e00到\u9fa5
        p2 = re.compile(r'[^\u4e00-\u9fa5]')
        result = " ".join(p2.split(line)).strip()
        cutline = jieba.cut(result,cut_all=False)
        seq_cut = " ".join(cutline)
        seq_cut = re.split(r'\s+', seq_cut)
        new_ = []
        for i in seq_cut:
            if i not in stop_words:
                new_.append(i)
                new_.append(' ')
        new_ = ''.join(new_)
        new_seg.append(new_)
    return new_seg





#提取关键词
def word_tf(df,n=1000):
    #将文本中的词语转换为词频矩阵
    vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+',max_features=n)
    #计算个词语出现的次数
    word_sum = vectorizer.fit_transform(df).toarray().sum(axis=0)
    #获取词袋中所有文本关键词
    word_dict = vectorizer.vocabulary_
    word_dict_sort = sorted(word_dict.items(),key = lambda x:x[1],reverse = False)
    word_name = []
    for i in word_dict_sort:
        word_name.append(i[0]) 
    d = {key: value for (key, value) in zip(word_name,word_sum)}
    d_sort = sorted(d.items(),key = lambda x:x[1],reverse = True)
    d_cipin = []
    for i in d_sort:
        d_cipin.append(i[0])
    d_cloud = ' '.join(d_cipin)
    return d_cloud,d_sort

def word_tfidf(df,n=1000):
    #将文本中的词语转换为词频矩阵
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+',max_features=n)
    #计算个词语出现的次数
    word_sum = vectorizer.fit_transform(df).toarray().sum(axis=0)
    #获取词袋中所有文本关键词
    word_dict = vectorizer.vocabulary_
    word_dict_sort = sorted(word_dict.items(),key = lambda x:x[1],reverse = False)
    word_name = []
    for i in word_dict_sort:
        word_name.append(i[0]) 
    d = {key: value for (key, value) in zip(word_name,word_sum)}
    d_sort = sorted(d.items(),key = lambda x:x[1],reverse = True)
    d_cipin = []
    for i in d_sort:
        d_cipin.append(i[0])
    d_cloud = ' '.join(d_cipin)
    return d_cloud,d_sort

def creat_wordcloud(word_cloud,picpath):
    #创建词云
    back_color = imread(picpath)  # 解析该图片
    wc = WordCloud(background_color='white',  # 背景颜色
               max_words=1000,  # 最大词数
               mask= back_color,  # 以该参数值作图绘制词云，这个参数不为空时，width和height会被忽略
               min_font_size=15,
               max_font_size=200,  # 显示字体的最大值
               font_path="C:/Windows/Fonts/fzfysjw.ttf",
               random_state=42,  # 为每个词返回一个PIL颜色
               width=10000,  # 图片的宽
               height=8600  #图片的长
               )
    wc.generate(word_cloud)
    # 基于彩色图像生成相应彩色
    image_colors = ImageColorGenerator(back_color)
    # 显示图片
    plt.imshow(wc)
    # 关闭坐标轴
    plt.axis('off')
    # 绘制词云
    plt.figure()
    plt.imshow(wc.recolor(color_func=image_colors))
    plt.axis('off')
    # 保存图片
    #plt.show()
    #wc.to_file('19th.png')

def review_equal(df1, df2):
    same = []
    for i in df1:
        for j in df2:
            if i == j:
                same.append(i)
    return same 
    




df = pd.read_csv('D:\\py\\emotion\\data\\data.csv')
df0 = list(df.loc[df['label'] == 0]['info'])
df1 = list(df.loc[df['label'] == 1]['info'])
df2 = list(df.loc[df['label'] == 2]['info'])
df3 = list(df.loc[df['label'] == 3]['info'])
df4 = list(df.loc[df['label'] == 4]['info'])
df5 = list(df.loc[df['label'] == 5]['info'])

'''

0 代表 Irritability
1 代表 Afraid
2 代表 Angry
3 代表 Happy
4 代表 Frustrated
5 代表 Help

'''
'0-4 same'

s0_4 = review_equal(df0, df4)
s0_1 = review_equal(df0, df1)
s0_2 = review_equal(df0, df2)
s1_2 = review_equal(df1, df2)
s1_4 = review_equal(df1, df4)
s5_1 = review_equal(df5, df1)
s5_2 = review_equal(df5, df2)
s5_4 = review_equal(df5, df4)

df_0 = Word_cut(df0)
word_cloud, d_sort = word_tf(df_0, n=1000)
picpath = 'timg.jpg'
#creat_wordcloud(word_cloud,picpath)

