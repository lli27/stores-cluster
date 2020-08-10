# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2019/10/28 14:20
@Desc   : 自己实现改编版kmeans算法
"""
import numpy as np
import logging
from itertools import repeat
from sklearn.utils.extmath import stable_cumsum
from multiprocessing.dummy import Pool


class Kmedoids():
    """
    改编版Kmeans聚类，不是直接计算类的均值作为中心点，而是遍历类中的每个点，选择其他点到该点的平均距离最小的点作为中心点。
    即k-means的中心点是欧式空间中任意一个连续的值，不一定在数据点中，但kmedoids是选择类中的点，该点一定在数据点中。
    k-means与k-mediods相当于均值与中位数的区别。
    直接输入距离矩阵，聚类数目，最大迭代次数，即可输出聚类标签，中心点
    """

    def __init__(self, n_clusters=2, n_init=10, max_iter=100, random_state=None, init_size=None, n_jobs=4, method="kmeans++"):
        """
        初始化参数
        :param n_clusters: 聚类数目
        :param max_iter: 最大迭代次数
        :param random_state: 随机数
        """
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.method = method
        self.rng = np.random.RandomState(random_state)  # 定义一个伪随机数生成器
        self.init_size = init_size
        self.n_jobs = n_jobs
        logging.basicConfig(level=logging.DEBUG,
                            filename='log/' + self.__class__.__name__ + ".log",
                            filemode='a',
                            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                            )

    def _init_centroids(self,X):
        n_samples = X.shape[0]  # 样本点的数目
        if self.init_size is not None and self.init_size < n_samples:
            if self.init_size < self.n_clusters:
                logging.warning("init_size={} should be larger than n_clusters={}. Setting it to 3*n_clusters".format(self.init_size, self.n_clusters))
                self.init_size = min(3*self.n_clusters,n_samples)
            init_indices = self.rng.randint(0,n_samples,self.init_size)
            X = X[np.ix_(init_indices,init_indices)]
        elif self.init_size is None:
            self.init_size = n_samples//3
            init_indices = self.rng.randint(0,n_samples,self.init_size)
            X = X[np.ix_(init_indices, init_indices)]
        elif n_samples < self.n_clusters:
            logging.error("n_samples={} should be larger than k={}".format(n_samples, self.n_clusters))
            raise ValueError("n_samples={} should be larger than k={}".format(n_samples, self.n_clusters))
        if isinstance(self.method, str) and self.method == 'kmeans++':
            cent_idx = self._kmedoids_init(X)
        elif isinstance(self.method, str) and self.method == 'random':
            cent_idx = self.rng.choice(n_samples, replace=False, size=self.n_clusters)
        elif hasattr(self.method, '__array__'):
            cent_idx = np.array(self.method, dtype=X.dtype)
        return cent_idx

    def _kmedoids_init(self, X):
        """
        kmeans++算法：寻找初始中心点
        :param X: 距离矩阵
        :return: 初始中心点
        """
        n_samples = X.shape[0]  # 样本点的数目
        n_local_trials = 2 + int(np.log(self.n_clusters))
        cent_idx = np.empty(shape=self.n_clusters, dtype=int)
        cent_idx[0] = self.rng.choice(n_samples, replace=False, size=1)  # 第一个中心点随机选取
        # 计算其他点到第一个中心点的距离
        closest_dist_sq = X[cent_idx[0], :]
        current_pot = closest_dist_sq.sum()
        # 生成剩下的n_clusters-1个聚类中心点
        for c in range(1, self.n_clusters):
            # 根据每个样本与当前已有聚类中心的最短距离，随机选择n_local_trials个待选中心点，距离越大，被选到的概率越大
            rand_vals = self.rng.random_sample(n_local_trials) * current_pot
            candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
            best_candidate = None
            best_pot = None
            best_dist_sq = None
            # 第二步：遍历n_local_trials个待选中心点，计算每个样本与当前已有聚类中心的最短距离，并求和。选择使该距离和最小的待选中心点作为下一个新的聚类中心点
            for candidate_id in candidate_ids:
                new_dist_sq = np.minimum(closest_dist_sq, X[candidate_id, :])  # 计算每个样本与当前已有聚类中心的最短距离
                new_pot = new_dist_sq.sum()  # 求和
                # 选择使该距离和最小的待选中心点
                if (best_candidate is None) or (new_pot < best_pot):
                    best_candidate = candidate_id
                    best_pot = new_pot
                    best_dist_sq = new_dist_sq
            cent_idx[c] = best_candidate
            current_pot = best_pot
            closest_dist_sq = best_dist_sq
        return cent_idx


    def _assign_clusters(self, X, cent_idx):
        """
        将剩下的点分配给距离它最近的中心点
        :param X: 距离矩阵
        :param cent_idx: 中心点
        :return: 类别标签，中心点，类间距离和（类间距离和越小越好）
        """
        membs = np.argmin(X[cent_idx, :], axis=0)
        inertia = 0
        for value in range(self.n_clusters):
            memb = np.where(membs == value)[0]
            inertia += X[np.ix_([cent_idx[value]], memb)].sum()
        return membs, cent_idx, inertia

    def _update_centers(self, X, membs, cent_idx):
        """
        更新中心点, 用类的中位数作为新的中心点
        :param X: 距离矩阵
        :param membs: 类别标签
        :param cent_idx: 上一次迭代的中心点
        :return: 新的中心点,上一次中心点
        """
        cent_idx_old = cent_idx.copy()
        for value in range(self.n_clusters):
            memb = np.where(membs == value)[0]
            if memb.shape[0] == 0:  # 如果这个类为空，则随机选择一个点放在这个类中
                memb = np.random.choice(X.shape[0], size=1)
            dist = np.mean(X[np.ix_(memb, memb)], axis=0)
            cent_idx[value] = memb[np.argmin(dist)]
        return cent_idx, cent_idx_old

    def _kmedoids_single(self, X):
        """
        迭代1.将剩下的点分配给距离它最近的中心点 2.更新中心点, 用类的中位数作为新的中心点
        直到类的中心点与类的中位数重合，或者达到最大迭代次数时，停止迭代。
        :param X: 距离矩阵
        :return: 聚类标签，中心点
        """
        cent_idx = self._init_centroids(X)
        for i in range(self.max_iter):
            membs, cent_idx, inertia = self._assign_clusters(X, cent_idx)
            cent_idx, cent_idx_old = self._update_centers(X, membs, cent_idx)
            if i%200 == 0 and i>0:
                if set(cent_idx) == set(cent_idx_old_cicle):
                    logging.info("iteration: {}, inertia: {}, center: {}, early stop".format(i, inertia, cent_idx))
                    break
            if i % 100 == 0 and i>0:
                cent_idx_old_cicle = cent_idx.copy()
                logging.info("iteration: {}, inertia: {}, center: {}".format(i, inertia, cent_idx))  # 每100次打印一次log
        membs, cent_idx, inertia = self._assign_clusters(X, cent_idx)
        return membs, cent_idx, inertia

    def _kmedoids(self, X):
        """
        选择不同的初始中心点多次运行kmedoids算法，选择最佳的中心点
        :param X:
        :return: 最佳聚类标签，最佳中心点，类间距离和
        """
        best_membs, best_inertia, best_cent_idx = None, None, None
        pool = Pool(self.n_jobs)
        result_list = pool.map(self._kmedoids_single, repeat(X,self.n_init))
        pool.close()
        pool.join()
        for it in result_list:
            if best_inertia is None or it[2] < best_inertia:
                best_membs = it[0].copy()
                best_cent_idx = it[1].copy()
                best_inertia = it[2]
                logging.info("best_inertia: {}".format(best_inertia))
        return best_membs, best_cent_idx, best_inertia

    def fit(self, X):
        """
        拟合，返回标签和中心点
        :param X: 距离矩阵
        :return: 标签，中心点，类间距离和
        """
        self.labels_, self.centers_, self.inertia = self._kmedoids(X)
        return self.labels_, self.centers_, self.inertia
