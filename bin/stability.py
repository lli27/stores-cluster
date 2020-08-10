# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2019/11/13 14:44
@Desc   : 计算稳定性
"""
import pandas as pd

for i in ['0','1','2','3','4']:
    d1=pd.read_csv(r"data_compare\2017-{}-country.csv".format(i))
    d1.columns=['str_code','label']
    d1['str_code']=d1['str_code'].astype('str')

    d2=pd.read_csv(r"data_compare\2018-{}-country.csv".format(i))
    d2.columns=['str_code','label']
    d2['str_code']=d2['str_code'].astype('str')

    t1,t2 = {},{}
    for index,item in d1.iterrows():
        t1[item['str_code']] = set(d1.loc[d1['label'] == item['label'], 'str_code'].values)
    for index,item in d2.iterrows():
        t2[item['str_code']] = set(d2.loc[d2['label'] == item['label'], 'str_code'].values)
    score = 0
    for key,values in t1.items():
        score += len(t2[key]&values)/len(t2[key]|values)
    print("score:%.2f"%(score/len(d1)))


