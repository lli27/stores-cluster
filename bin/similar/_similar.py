# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2019/5/27 10:19
@Desc   : 计算门店相似度，基于集合的距离度量方法。 门店A,B的相似度=A,B店爆款的交集/A店的爆款和B店款的交集，直观理解就是，在A店卖爆的款，在B店也爆卖的概率。
考虑到如果是热销款，则在大部分门店中，它的销售性质可能都为爆畅，故不能通过热销款来说明两家门店的相似性。故在计算A,B店爆款的交集时，对每个款乘以一个权重。权重系数
的计算方式为：1/log(m), m为该款成为爆畅款的门店数目。
相比无权重版，测试发现，它对相似门店的分配更均匀，即找出来的相似门店更具个性化。如：我们限定范围为，杭州市，寻找相似门店。看TOP1的门店，及对应数目。它的方差会更小。即
个别门店成为其他门店TOP1的次数减少。
"""

import pandas as pd
import numpy as np
import datetime
import sqlalchemy as sql
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import spatial
from abc import ABC, abstractmethod


def jaccard_union_weight(u, v, weight, icon):
    """
    计算分母，并加权
    :param u:
    :param v:
    :param weight: 权重系数
    :return: float
    """
    unequal_nonzero = np.bitwise_or(u, v).astype('float64')
    if icon:
        unequal_nonzero *= weight
    a = np.double(unequal_nonzero.sum())

    return a


def jaccard_inter_weight(u, v, weight, icon):
    """
    计算分子，并加权
    :param u:
    :param v:
    :param weight: 权重系数
    :return: float
    """
    unequal_nonzero = np.bitwise_and(u, v).astype('float64')
    if icon:
        unequal_nonzero *= weight
    a = np.double(unequal_nonzero.sum())

    return a


def itm_weight(df, classes):
    """
    计算每个款的权重
    :param df:
    :param classes: 款
    :return: list
    """
    itm_weight_dic = dict.fromkeys(classes, 0)
    for index, record in df.iterrows():
        for sty_code in record["sty_code_list"].split(","):
            itm_weight_dic[sty_code] += 1
    return [1 / np.log2(i) if i > 1 else 0 for i in itm_weight_dic.values()]


def distance(distrib_code, str_code, sty_code_label,sty_code_hot_label, itm_weight_list):
    # 计算距离，距离越小，即相交的款数越多，门店越相似
    k = 0
    m = len(sty_code_hot_label)
    dist = np.empty((m * (m - 1)) // 2, dtype=np.double)
    for i in range(0, m - 1):
        for j in range(i + 1, m):
            dist1 = jaccard_inter_weight(
                sty_code_hot_label[i], sty_code_hot_label[j], itm_weight_list, 1
            )
            dist2 = jaccard_union_weight(sty_code_hot_label[i], sty_code_hot_label[j], itm_weight_list, 1)
            if dist1==0:
                dist[k] = 1e10
            else:
                dist[k] = dist2 / dist1
            k += 1
    dist = spatial.distance.squareform(dist)
    return distrib_code, str_code, dist


def distrib(df):
    """
    根据分区计算门店距离
    :return: 返回一个生成器
    """
    # 店铺爆畅款存储在data_hot中，店铺卖过的所有款存储在data中
    sty_code_label = []
    distrib_set = df["distrib_code"].unique()
    for distrib in distrib_set:
        # 将店铺卖过的款作为店铺的标签，并进行数值化
        df_distrib = df[df["distrib_code"].isin([distrib])]
        df_distrib.sort_values(by=['str_code'], axis=0, inplace=True)
        distrib_code = df_distrib["distrib_code"].values
        str_code = df_distrib["str_code"].values
        sty_code_list = df_distrib["sty_code_list"].str.split(",")
        mlb = MultiLabelBinarizer()
        sty_code_hot_label = mlb.fit_transform(sty_code_list)
        # 计算商品权重，即商品在多少家门店成为过爆款
        itm_weight_list = itm_weight(df_distrib, mlb.classes_)
        yield distance(distrib_code, str_code, sty_code_label,sty_code_hot_label, itm_weight_list)


def country(df):
    """
    计算全国的相似门店
    :param df:
    :return: 返回一个生成器
    """
    data_hot = df[df["if_hot_itm"].isin(["1"])]
    data = df[df["if_hot_itm"].isin(["0"])]
    data_hot.sort_values(by=['str_code'], axis=0, inplace=True)
    data.sort_values(by=['str_code'], axis=0, inplace=True)
    distrib_code = data["distrib_code"].values
    # 记录门店编码
    str_code = data['str_code'].values
    # 将店铺卖过的款作为店铺的标签，并进行数值化
    sty_code_list = data["sty_code_list"].str.split(",")
    mlb = MultiLabelBinarizer()
    sty_code_label = mlb.fit_transform(sty_code_list)
    # 对店铺的爆畅款也进行数值化
    sty_code_hot_list = data_hot["sty_code_list"].str.split(",")
    sty_code_hot_label = mlb.transform(sty_code_hot_list)
    # 计算商品权重，即商品在多少家门店成为过爆款
    itm_weight_list = itm_weight(df, mlb.classes_)
    yield distance(distrib_code, str_code, sty_code_label,sty_code_hot_label, itm_weight_list)


class similar(ABC):
    """
    定义抽象类，fit方法需要在具体类中实现
    """

    def __init__(self,start_date,end_date):
        """
        建立数据库连接
        """
        connect_string = "" # 数据库账号/密码
        self.sql_engine = sql.create_engine(connect_string)
        self.end_date = end_date # datetime.date.today().strftime("%Y%m%d")
        self.start_date = start_date# (datetime.date.today()+datetime.timedelta(days=-365)).strftime("%Y%m%d")


    def execute_sql(self, sql_cmd):
        """
        执行sql语句
        :return: df
        """
        df = pd.read_sql(sql=sql_cmd, con=self.sql_engine)
        return df

    @abstractmethod
    def fit(self):
        pass


class country_similar(similar):
    """
    继承自抽象类的实体类，需要实现fit方法
    """

    def fit(self):
        sql_cmd = "" # 取店款销售性质的数据
        df = self.execute_sql(sql_cmd)
        return country(df)


class distrib_similar(similar):
    """
    继承抽象类的实体类，需要实现fit方法
    """

    def fit(self):
        sql_cmd = "" # 取店款销售性质的数据
        df = self.execute_sql(sql_cmd)
        return distrib(df)
