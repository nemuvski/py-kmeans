# -*- coding: utf-8 -*-

"""
デモ No.1
自作のk-meansモジュールを利用してみる
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def extract_patches(X, patch_size, num_patches, shuffle=False):
    """正方形のパッチを切り出して、返却する (シャッフル可能)

    Parameters
    ----------
    X: 画像配列 (N x height x width x rgb)
    patch_size: パッチの大きさ
    num_patches: 画像1枚から切り出すパッチの枚数
    shuffle=False: 返却するパッチ群をシャッフルするか

    Returns
    -------
    パッチ群
    """
    patches = []
    image_size = X.shape[1]  # X.shape[1:-1]でもよいが、パッチは正方形なので1つだけでよい

    for x in X:
        positions = np.random.randint(0, image_size - patch_size,
                                      (num_patches, 2))
        for p in positions:
            patches.append(x[p[0]:p[0] + patch_size,
                             p[1]:p[1] + patch_size,
                             :].ravel())
    patches = np.array(patches)
    if shuffle:
        np.random.shuffle(patches)
    return patches


def preprocessing(X):
    """前処理 ここでは標準化と白色化(whiten)を施す

    Parameters
    ----------
    X: ベクトル群

    Returns
    -------
    前処理後のベクトル群　
    """
    # 標準化
    X_mean = X.mean(axis=0).reshape((1, -1))
    X_std = X.std(axis=0, ddof=1).reshape((1, -1))
    X = (X - X_mean) / X_std
    del X_mean, X_std

    # 白色化
    X_cov = np.cov(X, rowvar=True, ddof=1)
    U, S, _ = np.linalg.svd(X_cov)
    eps = 1e-1
    P = np.dot(U, np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + eps)),
               U.T)))

    return np.dot(P, X)


def plot_centroids(centroids, patch_size):
    """パッチ群をクラスタリングした結果を表示

    Parameters
    ----------
    centroids: パッチ群をクラスタリングした結果
    patch_size: パッチの大きさ

    Returns
    -------
    なし
    """
    k = centroids.shape[0]
    for i, centroid in enumerate(centroids):
        p = (centroid - centroid.min()) / (centroid.max() - centroid.min())
        p = p.reshape((patch_size, patch_size, 3))
        plt.subplot(1, k, i+1)
        plt.imshow(p)
        plt.axis('off')
    plt.savefig('chart/rep/centroids.png')

    # 各チャンネルの内容
    for i, centroid in enumerate(centroids):
        p = (centroid - centroid.min()) / (centroid.max() - centroid.min())
        base_idx = 1 + (3 * i)
        p = p.reshape((patch_size, patch_size, 3))
        for ch in range(3):
            plt.subplot(k, 3, base_idx + ch)
            plt.imshow(p[:, :, ch])
            plt.axis('off')
    plt.savefig('chart/rep/centroids_ch.png')


def plot_fig(X, belongs, centroids):
    plt.figure()

    colors = [
        'crimson',
        'dodgerblue',
        'seagreen',
        'gold',
        'purple'
    ]

    # データ点
    for i, b in enumerate(belongs):
        plt.scatter(X[i, 0], X[i, 1], c=colors[int(b)], marker='o')

    # クラスタ代表点
    for i, c in enumerate(centroids):
        plt.scatter(c[0], c[1], c=colors[i], marker='o',
                    linewidths=1, edgecolors='black')

    # plt.show()
    plt.savefig('chart/rep/demo.png')


def main():
    sys.path.append(os.path.join(os.getcwd(), 'src', 'lib'))
    from kmeans import KMeans

    # シード値
    seed_value = 101244
    np.random.seed(seed=seed_value)

    # データ準備
    data = np.load('dataset/mini-cifar10.npy')

    # パッチを切り出し (正方形)
    patch_size = 6  # 切り出されるサイズは (patch_size x patch_size x 3)
    num_patches = 3  # 画像1枚から何枚のパッチを切り出すか
    patches = extract_patches(data, patch_size, num_patches, shuffle=True)
    patches = patches.astype(np.float32)  # 演算のために実数に変換しておく
    del data

    # 前処理
    patches = preprocessing(patches)

    # kmeansモデルの設定
    km = KMeans(seed_value=seed_value)

    # クラスタリング
    k = 5
    iters = 50
    centroids, belongs = km.fit(patches, k, iters)

    # 可視化
    plot_centroids(centroids, patch_size)
    plot_fig(patches, belongs, centroids)


if __name__ == '__main__':
    main()
