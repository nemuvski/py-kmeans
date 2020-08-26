# -*- coding: utf-8 -*-

"""
デモ No.0
自作のk-meansモジュールを利用してみる
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def plot_fig(X, belongs, centroids):
    plt.figure()

    colors = [
        'crimson',
        'dodgerblue',
        'seagreen',
        'gold'
    ]

    # データ点
    for i, b in enumerate(belongs):
        plt.scatter(X[i, 0], X[i, 1], c=colors[int(b)], marker='o')

    # クラスタ代表点
    for i, c in enumerate(centroids):
        plt.scatter(c[0], c[1], c=colors[i], marker='o',
                    linewidths=1, edgecolors='black')

    # plt.show()
    plt.savefig('chart/basic/demo.png')


def main():
    sys.path.append(os.path.join(os.getcwd(), 'src', 'lib'))
    from kmeans import KMeans

    # シード値設定, 乱数生成器を作成
    seed_value = 9239
    random_gen = np.random.RandomState(seed_value)  # main()で使う用
    # k-meansのインスタンスの作成
    km = KMeans(seed_value=seed_value)

    # サンプルデータ
    X = random_gen.randint(0, 100, (500, 2))

    # クラスタリング
    k = 4
    iters = 20
    centroids, belongs = km.fit(X, k, iters)

    # プロット
    plot_fig(X, belongs, centroids)


if __name__ == '__main__':
    main()
