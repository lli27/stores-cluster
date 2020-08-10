# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2019/11/14 20:34
@Desc   : 新店分组
"""
import pandas as pd
import sqlalchemy as sql

def new_str_cluster(df):
    df['str_code'] = df['str_code'].astype('str')
    df['label'] = df['label'].astype('str')
    connect_string = "" # 数据库账号/密码
    sql_engine = sql.create_engine(connect_string)
    sql_cmd = "" # 取门店基本信息数据
    df_full = pd.read_sql(sql=sql_cmd, con=sql_engine)
    df_set = set(df['str_code'].unique())
    df_full_set = set(df_full['str_code'].unique())
    df_new_set = df_full_set-df_set
    df = pd.merge(left=df_full, right=df,on="str_code",how="right")
    label_new = []
    for i in df_new_set:
        item = df_full.loc[df_full['str_code']==i]
        df_label = df.loc[(df['city_name']==item['city_name'].values[0]) & (df['cbd_type_code'] == item['cbd_type_code'].values[0])]
        d_city_cbd = pd.DataFrame(df_label['label'].value_counts())
        df_label = df.loc[(df['city_name']==item['city_name'].values[0])]
        d_city = pd.DataFrame(df_label['label'].value_counts())
        df_label = df.loc[(df['prov_name']==item['prov_name'].values[0]) & (df['cbd_type_code'] == item['cbd_type_code'].values[0])]
        d_cbd = pd.DataFrame(df_label['label'].value_counts())
        df_label = df.loc[(df['prov_name'] == item['prov_name'].values[0])]
        d_prov = pd.DataFrame(df_label['label'].value_counts())
        if len(d_city_cbd) != 0:
            label_new.append([i, d_city_cbd.index[0]])
        elif len(d_city) != 0:
            label_new.append([i, d_city.index[0]])
        elif len(d_cbd) != 0:
            label_new.append([i, d_cbd.index[0]])
        elif len(d_prov) != 0:
            label_new.append([i, d_prov.index[0]])
        else:
            print(item)
    label_new = pd.DataFrame(label_new, columns=['str_code','label'])
    label_new = pd.merge(left=df_full,right=label_new,on="str_code",how="right")
    df = pd.concat([df,label_new])
    df = df[['str_code','label']]
    df.columns = ['分区编码','分区名称','省份名称','城市名称','商圈类型','门店编码','门店名称','分组标签']
    return df

with pd.ExcelWriter(r"data_compare\全国相似店分组数据.xls") as writer:
    for j in [2017,2018]:
        for i in [0,1,2,3,4]:
            df = pd.read_csv(r"data_compare\{}-{}-country.csv".format(j,i))
            df = new_str_cluster(df)
            if i==0:
                df.to_excel(writer, sheet_name='{}年全年'.format(j), encoding="utf-8", index=False)
            else:
                df.to_excel(writer,sheet_name='{}年{}季度'.format(j,i),encoding="utf-8", index=False)
