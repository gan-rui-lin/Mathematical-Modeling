import numpy as np
import pandas as pd


def entropy_weight(df: pd.DataFrame, cost_class=None) -> tuple[dict, pd.DataFrame]:
    """
    熵权法, 通过计算各指标的熵值来确定其权重
    参数：
        df: 输入的数据框，包含各个方案的指标值
        cost_class: 需要转化为成本型的指标列表
    返回值：
        权重字典和综合评价值数据框
    """
    if cost_class is None:
        cost_class = []

    # 归一化，可选
    normalized_data = df.copy()
    for col in df.columns:
        col_max = df[col].max()

        if col in cost_class:
            # 对于极小型指标（如费用），使用公式: b_ij = 1 - a_ij / a_j^max
            normalized_data[col] = 1 - df[col] / col_max
        else:
            # 对于极大型指标，使用公式: b_ij = a_ij / a_j^max
            normalized_data[col] = df[col] / col_max

    # 熵权法主流程
    # 1. 计算比重矩阵P
    P = normalized_data.div(normalized_data.sum(axis=0), axis=1)

    # 2. 计算信息熵
    n, _ = P.shape  # n个方案，m个指标
    k = 1 / np.log(n)  # 熵值系数

    entropy = {}
    for col in P.columns:
        # 避免对0取对数，用一个很小的正数替代0
        p_ln_p = P[col] * np.log(P[col] + 1e-20)
        entropy[col] = -k * p_ln_p.sum()

    # 3. 计算 j 项指标的变异系数
    diversity = {}
    for indicator, ent in entropy.items():
        diversity[indicator] = 1 - ent

    # 4. 计算权重
    total_diversity = sum(diversity.values())
    weights = {}
    for indicator, div in diversity.items():
        weights[indicator] = div / total_diversity

    # 5. 计算第 i 个评价综合评价值
    # 正则化后的评价值
    # 评价值正则与否会稍微影响排名
    s_value = normalized_data.dot(pd.Series(weights))

    rank_s_df = (
        s_value.rank(method="first", ascending=False).to_frame("rank").astype(int)
    )

    # 把 s_value 和 rank_s_df 拼接起来
    s_rank = pd.concat([s_value, rank_s_df], axis=1)
    s_rank.columns = ["value", "rank"]

    return weights, s_rank


if __name__ == "__main__":

    data = [
        [93, 66, 86, 88, 77, 71, 90, 94],
        [97, 99, 61, 61, 75, 87, 70, 70],
        [65, 99, 94, 71, 91, 86, 80, 93],
        [97, 79, 98, 61, 92, 66, 88, 69],
        [85, 92, 87, 63, 67, 64, 96, 98],
        [63, 65, 91, 93, 80, 80, 99, 74],
        [71, 77, 90, 88, 78, 99, 82, 68],
        [82, 97, 76, 73, 86, 73, 65, 70],
        [99, 92, 86, 98, 89, 83, 66, 85],
        [99, 99, 67, 61, 90, 69, 70, 79],
    ]

    column_names = ["A", "B", "C", "D", "E", "F", "G", "H"]

    index_names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]

    df = pd.DataFrame(data, columns=column_names, index=index_names)

    weights, s_rank = entropy_weight(df)

    print(weights)

    print(s_rank)
