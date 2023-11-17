import pandas as pd
from itertools import product
import math
import numpy as np
import random
from typing import Any, Dict, Iterable, List, Tuple, Optional


def generate_random_vector(ranges: List[Tuple[float, float]]) -> List[float]:
    """在指定范围内按均匀分布生成随机向量

    :param ranges: 每个维度的分布范围
    :type ranges: List[Tuple[float, float]]
    :return: [description]
    :rtype: List[float]
    """
    return [(end-begin) * np.random.random() + begin for begin, end in ranges]


def iter_blocks(n: int, fragments: List[List[Tuple[float, float]]]):
    """按切片迭代空间分块

    :param n: 迭代数
    :type n: int
    :param fragments: List[空间分块]
    :type fragments: List[List[Tuple[float, float]]]
    
    :yield: 空间分布范围
    :rtype: Iterator[List[Tuple[float, float]]]
    """
    iterator = product(*fragments)
    for i in range(n):
        yield next(iterator)
    

def ramdom_vectors_by_blocks(n: int, fragments: List[List[Tuple[float, float]]]) -> List[List[float]]:
    """按空间分块均匀分布生成随机向量

    :param n: 向量数
    :type n: int
    :param fragments: List[空间分块]
    :type fragments: List[List[Tuple[float, float]]]
    :return: List[随机向量]
    :rtype: List[List[float]]
    """
    return [generate_random_vector(block) for block in iter_blocks(n, fragments)]


def random_vectors_by_partition(n: int, dimension: int)  -> List[List[float]]:
    """生成随机向量组

    :param n: 生成向量数
    :type n: int
    :param dimension: 向量维度
    :type dimension: int
    :return: 随机向量组
    :rtype: List[List[float]]

    >>> [
            [0.09866185481764456, 0.24677911604333763, 0.2564142446564428, 0.3661132487605418]
            [0.1959593783451019, 0.22758231911002214, 0.49157089229502204, 0.9325823551379472]
            [0.13523454915437688, 0.2734214436483937, 0.7084943844489697, 0.287845955930896]
            [0.19115298679551979, 0.3999493111669893, 0.5274002205879706, 0.673410868393441]
        ]
    """
    part_counts = math.ceil((math.pow(n, 1/dimension)))
    sep = 1 /part_counts
    fragment_parts = [(i*sep, (i+1)*sep) for i in range(part_counts)]
    fragments = [fragment_parts] * dimension
    return ramdom_vectors_by_blocks(n, fragments)


def random_vectors_df_by_partition(n: int, dimension: int, columns=None) -> pd.DataFrame:
    """生成随机向量组DataFrame

    :param n: 生成向量数
    :type n: int
    :param dimension: 向量维度
    :type dimension: int
    :param columns: 每个维度命名, defaults to None， 默认为[x1, x2, x3, ...]
    :type columns: [type], optional
    :return: 随机向量组DataFrame
    :rtype: pd.DataFrame

    >>>          x1        x2        x3        x4
        0  0.147465  0.287701  0.402964  0.380287
        1  0.467898  0.096063  0.162431  0.563873
        2  0.025641  0.366551  0.869946  0.218722
        3  0.043672  0.379130  0.976107  0.684275
    """
    centers = random_vectors_by_partition(n, dimension)
    if not columns:
        columns = list(map(lambda x: f"x{x+1}", range(dimension)))
    return pd.DataFrame(centers, columns=columns)


def normal_distributed_vectors(center: List[float], numbers: int, loc=0.0, scale=1.0) -> np.ndarray:
    """基于中心点以正态分布生成n个随机样本点，并以二维数组返回

    :param center: 中心点
    :type center: List[float]
    :param numbers: 生成样本数
    :type numbers: int
    :param loc: 均值偏移, defaults to 0.0
    :type loc: float, optional
    :param scale: 分布方差, defaults to 1.0
    :type scale: float, optional
    :return: 样本点
    :rtype: np.ndarray[n, len(center)]
    """
    carray = np.array(center)
    diffs = np.random.normal(loc, scale, (numbers, len(center)))
    return diffs + carray


def seperated_normal_distributed_vectors(center: list, numbers: int, *scales):
    carray = np.array(center)
    assert len(center) == len(scales)
    diffs = np.array([np.random.normal(0, scale, size=numbers) for scale in scales]).T
    return diffs + carray
        

def labled_normal_distributed_vectors(label_counts: list, dimension: int, loc=0.0, scale=1.0):
    centers = random_vectors_by_partition(len(label_counts), dimension)
    results = []
    for numbers, center in zip(label_counts, centers):
        results.append(
            normal_distributed_vectors(center, numbers, loc, scale)
        )
    return centers, results


def labeled_normal_df(centers: pd.DataFrame, counts: List[float], labels: Optional[List[Any]]=None, loc=0.0, scale=1.0) -> pd.DataFrame:
    """生成带标签的随机样本点(DataFrame)

    :param centers: 中心样本点
    :type centers: pd.DataFrame
        >>>          x1        x2        x3        x4
            0  0.147465  0.287701  0.402964  0.380287
            1  0.467898  0.096063  0.162431  0.563873
            2  0.025641  0.366551  0.869946  0.218722
            3  0.043672  0.379130  0.976107  0.684275
    :param counts: 样本点个数(按分类)
    :type counts: List[float]
        >>> [10000, 10000, 20000, 15000]
    :param labels: 样本标签, defaults to None
    :type labels: List[Any], optional
        >>> [0, 1, 2, 3]
    :param loc: 均值偏移, defaults to 0.0
    :type loc: float, optional
    :param scale:  分布方差, defaults to 1.0
    :type scale: float, optional
    :return: 随机样本点
    :rtype: pd.DataFrame
        >>>             x1        x2        x3        x4  label
            0     0.035569  0.442760  0.086710 -0.010190      0
            1    -0.015042  0.002670  0.200440  0.630119      0
            2     0.488894  0.601210  0.292091  0.559484      0
            3     0.512226  0.048532  0.484755  0.590250      0
            4     0.156176  0.339996  0.589159  0.052632      0
            ...        ...       ...       ...       ...    ...
            5495  0.653751  0.206390  0.468440  0.440768      3
            5496  0.276424 -0.019905  0.491247  0.787640      3
            5497  0.754375  0.301892  0.739218  0.388729      3
            5498  0.274969  0.429347  0.287882  0.807518      3
            5499  0.727999  0.445471  0.364701  0.507376      3
    """
    if not labels:
        labels = list(range(len(centers)))
    assert len(centers) == len(counts) == len(labels), \
        "Size of centers, counts and labels should be equal to each other but current sizes are:" + \
            f"centers[{len(centers)}] counts[{len(counts)}] labels[{len[labels]}]"
    results = []
    for center, count, label in zip(centers.values, counts, labels):
        vectors = normal_distributed_vectors(center, count, loc, scale)
        data = pd.DataFrame(vectors, columns=centers.columns)
        data["label"] = label
        results.append(data)
    return pd.concat(results, ignore_index=True)


def labeled_normal_vectors(centers: List[List[float]], counts: List[float], labels: Optional[List[Any]]=None, loc=0.0, scale=1.0, shuffle=True) -> List[Dict]:
    """生成带标签的随机样本点, 用于spark生成。
    
    :param centers: 中心样本点
    :type centers: List[List[float]]
        >>> [
                [0.09866185481764456, 0.24677911604333763, 0.2564142446564428, 0.3661132487605418]
                [0.1959593783451019, 0.22758231911002214, 0.49157089229502204, 0.9325823551379472]
                [0.13523454915437688, 0.2734214436483937, 0.7084943844489697, 0.287845955930896]
                [0.19115298679551979, 0.3999493111669893, 0.5274002205879706, 0.673410868393441]
            ]
    :param counts: 样本点个数(按分类)
    :type counts: List[float]
        >>> [10000, 10000, 20000, 15000]
    :param labels: 样本标签, defaults to None
    :type labels: List[Any], optional
        >>> [0, 1, 2, 3]
    :param loc: [description], defaults to 0.0
    :type loc: float, optional
    :param scale: [description], defaults to 1.0
    :type scale: float, optional
    :param shuffle: [description], defaults to True
    :type shuffle: bool, optional
    :return: 随机样本点与标签
    :rtype: List[Dict]
        >>> [
                {
                    'features': [0.10315041281572704, 0.11452526519102485, 0.04221026601103271, 0.05012305708497982],
                    'label': 0
                },
                ...
            ]

    """
    if not labels:
        labels = list(range(len(centers)))
    assert len(centers) == len(counts) == len(labels), \
        "Size of centers, counts and labels should be equal to each other but current sizes are:" + \
            f"centers[{len(centers)}] counts[{len(counts)}] labels[{len(labels)}]"
    results = []
    for center, count, label in zip(centers, counts, labels):
        vectors = normal_distributed_vectors(center, count, loc, scale)
        results.extend([{"features": list(map(float, vector)), "label": label} for vector in vectors])
    if shuffle:
        random.shuffle(results)
    return results



def train(samples: pd.DataFrame):
    from sklearn import svm
    X = samples[samples.columns[:-1]].values
    y = samples["label"].values
    clf = svm.SVC()
    svc = clf.fit(X, y)
    return svc


def precision(svc, samples):
    X = samples[samples.columns[:-1]].values
    y = samples["label"].values
    prediction = svc.predict(X)
    result = prediction == y
    return result.sum()/len(result)


def estimate_scale(centers: list, counts: list, init_scale: float=1, expected_precision: float=0.9, diff: float=0.01):
    p = 0
    upper, lower = expected_precision+diff, expected_precision - diff
    scale = init_scale
    scale_high = None
    scale_low = 0
    center_df = pd.DataFrame(centers)
    while True:
        test_sample = labeled_normal_df(center_df, counts, scale=scale)
        samples = labeled_normal_df(center_df, counts, scale=scale)
        svc = train(samples)
        p = precision(svc, test_sample)
        print(scale, p)
        if p > upper:
            scale_low = scale
            if not scale_high:
                scale = scale * 2
            else:
                scale = (scale_high+scale) / 2
                
        elif p < lower:
            scale_high = scale
            scale = (scale_low+scale)/2
        else:
            
            return scale, p

