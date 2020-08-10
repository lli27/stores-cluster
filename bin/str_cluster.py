# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2019/10/29 10:44
@Desc   : 相似店聚类
"""
from similar import distrib_similar
from similar import country_similar
from cluster import Kmedoids
from sklearn.metrics import silhouette_score
from multiprocessing.dummy import Pool
from itertools import repeat
import pandas as pd
import os
import warnings

warnings.filterwarnings(action='ignore')


class str_cluster():
    def __init__(self, n_clusters_list=[2], n_init=10, max_iter=100, random_state=None, init_size=None, n_jobs=4, method="kmeans++",
                 store="country",start_date=None,end_date=None):
        if not os.path.isdir('log'):
            os.makedirs('log')
        if not os.path.isdir('data'):
            os.makedirs('data')
        self.n_clusters_list = n_clusters_list
        self.n_init = n_init
        self.max_iter = max_iter
        self.method = method
        self.random_state = random_state
        self.store = store
        self.init_size = init_size
        self.n_jobs = n_jobs
        self.start_date = start_date
        self.end_date = end_date

    def fit(self, num, D):
        kmediods = Kmedoids(n_clusters=num, n_init=self.n_init, max_iter=self.max_iter, n_jobs=self.n_jobs,
                            random_state=self.random_state)  # 轮廓系数越大，聚类效果越好，确定聚类数目
        return  num,kmediods.fit(D)
    def main(self):
        if self.store == "distrib":
            similar = distrib_similar(self.start_date,self.end_date)
        else:
            similar = country_similar(self.start_date,self.end_date)
        D_tuple = similar.fit()
        result = pd.DataFrame()
        for distrib_code, str_code, D in D_tuple:
            print(distrib_code[0])
            score_best = None
            num_best = None
            label_best = None
            center_best = None
            pool = Pool(self.n_jobs)
            result_list = pool.starmap(self.fit, zip(self.n_clusters_list,repeat(D)))
            for it in result_list:
                score = silhouette_score(D, it[1][0], metric="precomputed", random_state=1)
                if (score_best is None) or (score > score_best):
                    score_best = score
                    num_best = it[0]
                    label_best = it[1][0]
                    center_best = it[1][1]
            print("score_best: {}\nnum_best: {}".format(score_best, num_best))
            distrib_result = pd.DataFrame(label_best, index=[str_code], columns=["label"])
            result = result.append(distrib_result)
        return result

if __name__ == "__main__":
    cluster = str_cluster(n_clusters_list=[10, 12, 15], n_init=30, max_iter=30000, n_jobs=20, random_state=1,
                          store="distrib")
    # cluster = str_cluster()
    result = cluster.main()
    result.to_csv("data/test.csv", index=True, index_label=["str_code"])
