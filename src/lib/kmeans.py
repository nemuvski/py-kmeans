# -*- coding: utf-8 -*-

"""
k-means用のモジュール
"""

import sys
import numpy as np


class KMeans:
    def __init__(self, seed_value=None):
        """ランダム生成器の生成, クラスタ代表点変数の初期化

        Parameters
        ----------
        seed_value=None: シード値

        Returns
        -------
        なし
        """
        self.__random_generator = np.random.RandomState(seed_value)
        self.__centroids = None

    @property
    def centroids(self):
        """クラスタ代表点群を返却 (ゲッター)

        Parameters
        ----------
        なし

        Returns
        -------
        クラスタ代表点群 (fit()が未実行の場合はNone)
        """
        return self.__centroids

    def fit(self, X, k, iterations):
        """クラスタ代表点を決定

        Parameters
        ----------
        X: クラスタ代表点を決定する特徴ベクトル群
        k: クラスタ数 (1以上の整数)
        iterations: クラスタ代表点更新回数 (1以上の整数)

        Returns
        -------
        クラスタ代表点群と所属クラスタのフラグ
        """
        # 引数確認
        if k <= 0:
            print('クラスタ数は1以上の整数としてください')
            sys.exit(1)
        if k > X.shape[0]:
            print('クラスタ数はデータ数よりも小さい値としてください')
            sys.exit(1)
        if iterations <= 0:
            print('回数は1以上の整数としてください')
            sys.exit(1)
        # クラスタ代表点群を初期化
        self.__centroids = self.__random_generator.randn(k, X.shape[1])
        # 所属クラスタのフラグを初期化
        belongs = np.zeros((X.shape[0], 1), dtype=np.int)
        print('クラスタ数: {0}\n更新回数: {1}'.format(k, iterations))
        print('----------------')
        # クラスタリング
        for i in range(iterations):
            print('{0} / {1}'.format(i + 1, iterations))
            # 所属クラスタのフラグを更新
            self.__update_belongs(X, belongs)
            # 各クラスタ代表点のベクトルを更新
            self.__update_centroids(X, belongs)
        return (self.__centroids, belongs)

    def __update_belongs(self, X, belongs):
        """所属クラスタのフラグを更新

        Parameters
        ----------
        X: クラスタ代表点を決定する特徴ベクトル群
        belongs: 所属クラスタのフラグ (このメソッド内で更新される)

        Returns
        -------
        なし
        """
        for i, x in enumerate(X):
            belongs[i] = self.__get_cluster_index(x)

    def __get_cluster_index(self, x):
        """特徴ベクトルと各クラスタ代表点との距離を算出し、最小値をとるクラスタ番号を返却

        Parameters
        ----------
        x: 特徴ベクトル

        Returns
        -------
        クラスタ番号
        """
        return np.sum((x - self.__centroids) ** 2, axis=1).argmin()

    def __update_centroids(self, X, belongs):
        """各クラスタ代表点の更新

        Parameters
        ----------
        X: クラスタ代表点を決定する特徴ベクトル群
        belongs: 所属クラスタのフラグ

        Returns
        -------
        なし
        """
        for i in range(self.__centroids.shape[0]):
            index, _ = np.where(belongs == i)
            # クラスタに所属するデータが1つもない場合は計算しない
            if index.size > 0:
                self.__centroids[i] = X[index, :].mean(axis=0)
