import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from entropy_weight import entropy_weight

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def topsis_analysis(data, cost_class=None, scheme_names=None, indicator_names=None):
    """
    TOPSIS多准则决策分析，使用熵值法计算权重

    参数:
        data: 决策矩阵 (DataFrame 或 list)
        cost_class: 成本型指标列表
        scheme_names: 方案名称列表
        indicator_names: 指标名称列表

    返回:
        完整的TOPSIS分析结果字典
    """

    if cost_class is None:
        cost_class = []

    # 数据预处理
    if isinstance(data, list):
        if scheme_names is None:
            scheme_names = [f"方案{i+1}" for i in range(len(data))]
        if indicator_names is None:
            indicator_names = [f"指标{i+1}" for i in range(len(data[0]))]
        df = pd.DataFrame(data, index=scheme_names, columns=indicator_names)
    else:
        df = data.copy()

    print("=" * 60)
    print("【TOPSIS多准则决策分析】")
    print("=" * 60)

    print("\n原始决策矩阵:")
    print(df)

    # 步骤1: 使用熵值法计算权重
    print("\n" + "=" * 40)
    print("【步骤1: 熵值法计算指标权重】")
    print("=" * 40)

    weights, entropy_result = entropy_weight(df, cost_class)

    print("\n指标权重分布:")
    total_weight = 0
    for indicator, weight in weights.items():
        importance = "高" if weight > 0.2 else "中" if weight > 0.1 else "低"
        print(f"  {indicator}: {weight:.4f} ({importance}重要性)")
        total_weight += weight
    print(f"  权重总和: {total_weight:.4f}")

    # 步骤2: 数据标准化
    print("\n" + "=" * 40)
    print("【步骤2: 数据标准化处理】")
    print("=" * 40)

    # 对原始数据进行标准化（向量归一化）
    normalized_matrix = df.copy().astype(float)
    for col in df.columns:
        col_sum_squares = np.sqrt((df[col] ** 2).sum())
        normalized_matrix[col] = df[col] / col_sum_squares

    print("\n标准化决策矩阵:")
    print(normalized_matrix.round(4))

    # 步骤3: 构造加权标准化决策矩阵
    print("\n" + "=" * 40)
    print("【步骤3: 构造加权标准化决策矩阵】")
    print("=" * 40)

    weighted_matrix = normalized_matrix.copy()
    for col in normalized_matrix.columns:
        weighted_matrix[col] = normalized_matrix[col] * weights[col]

    print("\n加权标准化决策矩阵:")
    print(weighted_matrix.round(6))

    # 步骤4: 确定正理想解和负理想解
    print("\n" + "=" * 40)
    print("【步骤4: 确定理想解】")
    print("=" * 40)

    positive_ideal = {}  # 正理想解 (A+)
    negative_ideal = {}  # 负理想解 (A-)

    for col in weighted_matrix.columns:
        if col in cost_class:
            # 成本型指标: 最小值为正理想解
            positive_ideal[col] = weighted_matrix[col].min()
            negative_ideal[col] = weighted_matrix[col].max()
        else:
            # 效益型指标: 最大值为正理想解
            positive_ideal[col] = weighted_matrix[col].max()
            negative_ideal[col] = weighted_matrix[col].min()

    print("\n正理想解 (A+):")
    for indicator, value in positive_ideal.items():
        indicator_type = "成本型" if indicator in cost_class else "效益型"
        print(f"  {indicator} ({indicator_type}): {value:.6f}")

    print("\n负理想解 (A-):")
    for indicator, value in negative_ideal.items():
        indicator_type = "成本型" if indicator in cost_class else "效益型"
        print(f"  {indicator} ({indicator_type}): {value:.6f}")

    # 步骤5: 计算距离
    print("\n" + "=" * 40)
    print("【步骤5: 计算欧几里得距离】")
    print("=" * 40)

    distances_positive = {}  # 到正理想解的距离 D+
    distances_negative = {}  # 到负理想解的距离 D-

    for idx in weighted_matrix.index:
        # 计算到正理想解的距离
        dist_pos = 0
        for col in weighted_matrix.columns:
            dist_pos += (weighted_matrix.loc[idx, col] - positive_ideal[col]) ** 2
        distances_positive[idx] = np.sqrt(dist_pos)

        # 计算到负理想解的距离
        dist_neg = 0
        for col in weighted_matrix.columns:
            dist_neg += (weighted_matrix.loc[idx, col] - negative_ideal[col]) ** 2
        distances_negative[idx] = np.sqrt(dist_neg)

    print("\n各方案到正理想解的距离 (D+):")
    for scheme, distance in distances_positive.items():
        print(f"  {scheme}: {distance:.6f}")

    print("\n各方案到负理想解的距离 (D-):")
    for scheme, distance in distances_negative.items():
        print(f"  {scheme}: {distance:.6f}")

    # 步骤6: 计算相对贴近度
    print("\n" + "=" * 40)
    print("【步骤6: 计算相对贴近度】")
    print("=" * 40)

    closeness = {}
    for idx in weighted_matrix.index:
        closeness[idx] = distances_negative[idx] / (
            distances_positive[idx] + distances_negative[idx]
        )

    print("\n各方案的相对贴近度 (Ci):")
    for scheme, close in closeness.items():
        print(f"  {scheme}: {close:.6f}")

    # 步骤7: 排序
    print("\n" + "=" * 40)
    print("【步骤7: 最终排序结果】")
    print("=" * 40)

    sorted_schemes = sorted(closeness.items(), key=lambda x: x[1], reverse=True)

    print("\n排名  方案    相对贴近度   性能评价")
    print("-" * 40)
    for rank, (scheme, close) in enumerate(sorted_schemes, 1):
        performance = (
            "优秀"
            if close > 0.7
            else "良好" if close > 0.5 else "一般" if close > 0.3 else "较差"
        )
        print(f"{rank:2d}   {scheme:6s}  {close:.6f}     {performance}")

    # 返回完整结果
    results = {
        "original_data": df,
        "weights": weights,
        "entropy_result": entropy_result,
        "normalized_matrix": normalized_matrix,
        "weighted_matrix": weighted_matrix,
        "positive_ideal": positive_ideal,
        "negative_ideal": negative_ideal,
        "distances_positive": distances_positive,
        "distances_negative": distances_negative,
        "closeness": closeness,
        "ranking": sorted_schemes,
    }

    return results


def plot_topsis_results(results):
    """绘制TOPSIS分析结果的可视化图表"""

    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 权重分布图
    weights = results["weights"]
    indicators = list(weights.keys())
    weight_values = list(weights.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(indicators)))

    bars1 = ax1.bar(indicators, weight_values, color=colors)
    ax1.set_title("熵值法计算的指标权重分布", fontsize=14, fontweight="bold")
    ax1.set_ylabel("权重值")
    ax1.tick_params(axis="x", rotation=45)

    # 添加数值标签
    for bar, value in zip(bars1, weight_values):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    # 2. 原始数据雷达图
    df = results["original_data"]
    angles = np.linspace(0, 2 * np.pi, len(indicators), endpoint=False).tolist()
    angles += angles[:1]

    # 数据归一化到0-1范围用于雷达图
    radar_data = df.copy()
    for col in df.columns:
        radar_data[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    colors_radar = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57"]
    for i, scheme in enumerate(radar_data.index):
        values = radar_data.loc[scheme].tolist()
        values += values[:1]
        color = colors_radar[i % len(colors_radar)]
        ax2.plot(angles, values, "o-", linewidth=2, label=scheme, color=color)
        ax2.fill(angles, values, alpha=0.25, color=color)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(indicators)
    ax2.set_ylim(0, 1.1)
    ax2.set_title("各方案指标雷达图", fontsize=14, fontweight="bold")
    ax2.legend(
        loc="upper right",
        bbox_to_anchor=(1, 1),
        borderaxespad=0.2,
        fontsize=10,
        frameon=True,
    )
    ax2.grid(True)

    # 3. 距离对比图
    schemes = list(results["distances_positive"].keys())
    d_plus = list(results["distances_positive"].values())
    d_minus = list(results["distances_negative"].values())

    x = np.arange(len(schemes))
    width = 0.35

    bars2 = ax3.bar(
        x - width / 2,
        d_plus,
        width,
        label="到正理想解距离(D+)",
        color="#FF6B6B",
        alpha=0.8,
    )
    bars3 = ax3.bar(
        x + width / 2,
        d_minus,
        width,
        label="到负理想解距离(D-)",
        color="#4ECDC4",
        alpha=0.8,
    )

    ax3.set_xlabel("方案")
    ax3.set_ylabel("欧几里得距离")
    ax3.set_title("各方案到理想解的距离对比", fontsize=14, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(schemes)
    ax3.legend()

    # 添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars3:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # 4. 相对贴近度排序
    ranking = results["ranking"]
    schemes_sorted = [item[0] for item in ranking]
    closeness_sorted = [item[1] for item in ranking]

    # 颜色梯度：第一名最亮，最后一名最暗
    colors_rank = plt.cm.viridis(np.linspace(0.3, 1, len(schemes_sorted)))

    bars4 = ax4.bar(schemes_sorted, closeness_sorted, color=colors_rank)
    ax4.set_title("TOPSIS相对贴近度排序结果", fontsize=14, fontweight="bold")
    ax4.set_ylabel("相对贴近度 (Ci)")
    ax4.set_ylim(0, 1)

    # 添加数值标签和排名
    for i, (bar, closeness_val) in enumerate(zip(bars4, closeness_sorted)):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{closeness_val:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height / 2,
            f"第{i+1}名",
            ha="center",
            va="center",
            fontweight="bold",
            color="white",
        )

    plt.tight_layout()
    plt.show()


def generate_analysis_report(results):
    """生成详细的分析报告"""

    print("\n" + "=" * 60)
    print("【TOPSIS综合分析报告】")
    print("=" * 60)

    weights = results["weights"]
    ranking = results["ranking"]

    # 权重分析
    print(f"\n 权重分析:")
    print(f"   通过熵值法计算得出各指标权重分布：")
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for i, (indicator, weight) in enumerate(sorted_weights, 1):
        importance = (
            "关键"
            if weight > 0.3
            else "重要" if weight > 0.15 else "一般" if weight > 0.08 else "较低"
        )
        print(f"   {i}. {indicator}: {weight:.4f} ({importance}影响)")

    # 排序分析
    print(f"\n 排序分析:")
    print(f"   基于TOPSIS算法的最终排序结果：")
    for rank, (scheme, closeness) in enumerate(ranking, 1):
        performance = (
            "优秀"
            if closeness > 0.7
            else "良好" if closeness > 0.5 else "一般" if closeness > 0.3 else "较差"
        )
        print(f"   第{rank}名: {scheme} (Ci={closeness:.4f}, {performance})")

    # 决策建议
    best_scheme = ranking[0][0]
    best_closeness = ranking[0][1]
    worst_scheme = ranking[-1][0]
    worst_closeness = ranking[-1][1]

    print(f"\n 决策建议:")
    print(f"    推荐方案: {best_scheme}")
    print(f"     相对贴近度: {best_closeness:.4f}")
    print(f"     该方案在综合平衡所有指标后表现最优")

    # 关键发现
    max_weight_indicator = max(weights, key=weights.get)
    max_weight_value = weights[max_weight_indicator]

    print(f"   决定性指标: '{max_weight_indicator}' (权重: {max_weight_value:.4f})")
    print(f"   该指标对最终决策结果影响最大")

    gap = best_closeness - worst_closeness
    print(f"    方案差异度: {gap:.4f}")
    if gap > 0.5:
        print(f"     各方案差异明显，选择空间较大")
    elif gap > 0.2:
        print(f"     各方案有一定差异，需仔细比较")
    else:
        print(f"     各方案差异较小，可考虑其他因素")


if __name__ == "__main__":
    # 示例数据：飞机选型问题
    aircraft_data = [
        # [最大速度, 飞行半径, 最大负载, 费用, 可靠性, 灵敏度]
        [2.0, 1500, 20000, 5500000, 0.5, 1.0],  # A1
        [2.5, 2700, 18000, 6500000, 0.3, 0.5],  # A2
        [1.8, 2000, 21000, 4500000, 0.7, 0.7],  # A3
        [2.2, 1800, 20000, 5000000, 0.5, 0.5],  # A4
    ]

    aircraft_names = ["A1", "A2", "A3", "A4"]
    indicator_names = ["最大速度", "飞行半径", "最大负载", "费用", "可靠性", "灵敏度"]
    cost_indicators = ["费用"]  # 费用为成本型指标

    # 执行TOPSIS分析
    results = topsis_analysis(
        data=aircraft_data,
        cost_class=cost_indicators,
        scheme_names=aircraft_names,
        indicator_names=indicator_names,
    )

    # 生成可视化图表
    plot_topsis_results(results)

    # 生成分析报告
    generate_analysis_report(results)
