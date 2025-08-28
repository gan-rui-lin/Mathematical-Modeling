import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def gra(data, cost_class=None, weights=None, rho=0.5, verbose=True):
    """
    灰色关联度分析函数

    参数:
        data: DataFrame, 决策矩阵，行为方案，列为指标
        cost_class: list, 成本型指标列表，默认为None（所有指标视为效益型）
        weights: np.array, 各指标权重，默认为None（等权重）
        rho: float, 分辨系数，通常取0.5
        verbose: bool, 是否打印详细信息

    返回:
        dict: 包含以下内容的字典
            - 'correlation_coefficients': 关联系数矩阵
            - 'correlation_degrees': 各方案关联度
            - 'ranking': 排序结果
            - 'reference_series': 参考序列
            - 'weights': 使用的权重
    """

    if cost_class is None:
        cost_class = []

    # 数据预处理：标准化
    df = data.copy()

    if verbose:
        print("=" * 60)
        print("【灰色关联度分析(Grey Relational Analysis)】")
        print("=" * 60)
        print(f"\n原始数据矩阵 ({df.shape[0]}个方案, {df.shape[1]}个指标):")
        print(df)

    # 数据标准化处理
    for col in df.columns:
        if col in cost_class:
            # 成本型指标：越小越好
            df[col] = (df[col].max() - df[col]) / (df[col].max() - df[col].min())
        else:
            # 效益型指标：越大越好
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    if verbose:
        print(f"\n标准化后数据矩阵:")
        print(df.round(4))

    # 步骤1: 确定参考序列（理想方案）
    reference_sequence = []
    for col in df.columns:
        # 标准化后所有指标都是越大越好
        reference_sequence.append(df[col].max())

    reference_series = pd.Series(reference_sequence, index=df.columns, name="理想方案")

    if verbose:
        print(f"\n" + "=" * 40)
        print("【步骤1: 确定参考序列】")
        print("=" * 40)
        print(f"\n参考序列（理想方案）:")
        print(reference_series.round(4))

    # 步骤2: 计算灰色关联系数
    correlation_coefficients = pd.DataFrame(index=df.index, columns=df.columns)

    if verbose:
        print(f"\n" + "=" * 40)
        print("【步骤2: 计算灰色关联系数】")
        print("=" * 40)
        print(f"分辨系数 ρ = {rho}")
        print(
            "关联系数公式: γ(x₀(k), xᵢ(k)) = (min_min + ρ×max_max) / (Δᵢ(k) + ρ×max_max)"
        )

    # 计算所有绝对差值
    all_deltas = []
    for scheme in df.index:
        for indicator in df.columns:
            delta = abs(df.loc[scheme, indicator] - reference_series[indicator])
            all_deltas.append(delta)

    min_min = min(all_deltas)  # 最小的最小差
    max_max = max(all_deltas)  # 最大的最大差

    # 计算关联系数矩阵
    for scheme in df.index:
        if verbose:
            print(f"\n{scheme}:")

        for indicator in df.columns:
            # 计算绝对差值
            delta = abs(df.loc[scheme, indicator] - reference_series[indicator])

            # 计算关联系数
            if delta + rho * max_max == 0:
                coefficient = 1.0  # 避免除零
            else:
                coefficient = (min_min + rho * max_max) / (delta + rho * max_max)

            correlation_coefficients.loc[scheme, indicator] = coefficient

            if verbose:
                print(f"  {indicator}: Δ={delta:.4f}, γ={coefficient:.4f}")

    if verbose:
        print(f"\n完整关联系数矩阵:")
        print(correlation_coefficients.round(4))

    # 步骤3: 计算灰色关联度
    if weights is None:
        # 等权重
        weights = np.array([1 / len(df.columns)] * len(df.columns))
    else:
        # 检查权重有效性
        if len(weights) != len(df.columns):
            raise ValueError(
                f"权重数组长度({len(weights)})与指标数量({len(df.columns)})不匹配"
            )
        if not np.isclose(weights.sum(), 1.0):
            print(f"警告: 权重总和为{weights.sum():.4f}，不等于1，将进行归一化处理")
            weights = weights / weights.sum()

    if verbose:
        print(f"\n" + "=" * 40)
        print("【步骤3: 计算灰色关联度】")
        print("=" * 40)
        print(f"使用的权重:")
        for i, (indicator, weight) in enumerate(zip(df.columns, weights)):
            print(f"  {indicator}: {weight:.4f}")
        print(f"权重总和: {weights.sum():.4f}")

    # 计算各方案的关联度（加权平均）
    correlation_degrees = correlation_coefficients.dot(weights)

    if verbose:
        print(f"\n各方案的灰色关联度:")
        for scheme, degree in correlation_degrees.items():
            print(f"  {scheme}: {degree:.6f}")

    # 步骤4: 排序分析
    sorted_schemes = correlation_degrees.sort_values(ascending=False)

    if verbose:
        print(f"\n" + "=" * 40)
        print("【步骤4: 排序分析】")
        print("=" * 40)
        print("最终排序结果:")
        print("排名  方案    关联度     优劣评价")
        print("-" * 35)

        for rank, (scheme, degree) in enumerate(sorted_schemes.items(), 1):
            if degree >= 0.8:
                evaluation = "优秀"
            elif degree >= 0.7:
                evaluation = "良好"
            elif degree >= 0.6:
                evaluation = "一般"
            else:
                evaluation = "较差"

            print(f"{rank:2d}   {scheme:6s}  {degree:.6f}   {evaluation}")

        best_scheme = sorted_schemes.index[0]
        best_degree = sorted_schemes.iloc[0]
        print(f"\n最优方案: {best_scheme} (关联度: {best_degree:.6f})")

    # 返回结果
    results = {
        "correlation_coefficients": correlation_coefficients,
        "correlation_degrees": correlation_degrees,
        "ranking": sorted_schemes,
        "reference_series": reference_series,
        "weights": weights,
        "standardized_data": df,
    }

    return results


def plot_gra_results(results):
    """绘制灰色关联度分析结果的可视化图表"""

    correlation_coefficients = results["correlation_coefficients"]
    sorted_schemes = results["ranking"]
    weights = results["weights"]
    df = results["standardized_data"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 关联度排序柱状图
    schemes_list = sorted_schemes.index.tolist()
    degrees_list = sorted_schemes.values.tolist()

    colors = plt.cm.viridis(np.linspace(0.3, 1, len(schemes_list)))
    bars1 = ax1.bar(schemes_list, degrees_list, color=colors)

    ax1.set_title("灰色关联度排序结果", fontsize=14, fontweight="bold")
    ax1.set_ylabel("关联度")
    ax1.set_ylim(0, 1)

    # 添加数值标签和排名
    for i, (bar, degree) in enumerate(zip(bars1, degrees_list)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{degree:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height / 2,
            f"第{i+1}名",
            ha="center",
            va="center",
            fontweight="bold",
            color="white",
        )

    # 2. 关联系数热力图
    sns.heatmap(
        correlation_coefficients.astype(float),
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax2,
        cbar_kws={"label": "关联系数"},
    )
    ax2.set_title("各方案-指标关联系数热力图", fontsize=14, fontweight="bold")
    ax2.set_xlabel("评价指标")
    ax2.set_ylabel("方案")

    # 3. 雷达图对比（前4名方案）
    top4_schemes = sorted_schemes.head(4).index
    angles = np.linspace(0, 2 * np.pi, len(df.columns), endpoint=False).tolist()
    angles += angles[:1]

    colors_radar = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
    for i, scheme in enumerate(top4_schemes):
        values = correlation_coefficients.loc[scheme].values.tolist()
        values += values[:1]
        ax3.plot(angles, values, "o-", linewidth=2, label=scheme, color=colors_radar[i])
        ax3.fill(angles, values, alpha=0.25, color=colors_radar[i])

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels([f"指标{i}" for i in range(1, len(df.columns) + 1)])
    ax3.set_ylim(0, 1)
    ax3.set_title("TOP4方案关联系数雷达图", fontsize=14, fontweight="bold")
    ax3.legend(loc="upper right", bbox_to_anchor=(1, 1))
    ax3.grid(True)

    # 4. 权重分布图
    bars4 = ax4.bar(
        range(len(weights)), weights, color=plt.cm.Set3(np.linspace(0, 1, len(weights)))
    )

    ax4.set_title("各指标权重分布", fontsize=14, fontweight="bold")
    ax4.set_ylabel("权重值")
    ax4.set_xticks(range(len(weights)))
    ax4.set_xticklabels([f"指标{i}" for i in range(1, len(weights) + 1)], rotation=45)

    # 添加数值标签
    for bar, value in zip(bars4, weights):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


def generate_gra_report(results):
    """生成灰色关联度分析详细报告"""

    sorted_schemes = results["ranking"]

    print("\n" + "=" * 60)
    print("【灰色关联度分析报告】")
    print("=" * 60)

    # 方案排序分析
    print(f"\n方案排序分析:")
    for rank, (scheme, degree) in enumerate(sorted_schemes.items(), 1):
        performance = (
            "优秀"
            if degree >= 0.8
            else "良好" if degree >= 0.7 else "一般" if degree >= 0.6 else "较差"
        )
        print(f"   第{rank}名: {scheme} (关联度: {degree:.6f}, {performance})")

    # 决策建议
    best_scheme = sorted_schemes.index[0]
    best_degree = sorted_schemes.iloc[0]

    print(f"\n决策建议:")
    print(f"   推荐方案: {best_scheme}")
    print(f"   关联度: {best_degree:.6f}")
    print(f"   该方案与理想方案最为接近")


if __name__ == "__main__":
    # 测试数据：供应商评价问题
    print("灰色关联度分析测试")
    print("=" * 50)

    # 创建测试数据
    data = {
        "产品质量": [0.83, 0.90, 0.99, 0.92, 0.87, 0.95],
        "产品价格": [326, 295, 340, 287, 310, 303],
        "地理位置": [21, 38, 25, 19, 27, 10],
        "售后服务": [3.2, 2.4, 2.2, 2.0, 0.9, 1.7],
        "技术水平": [0.20, 0.25, 0.12, 0.33, 0.20, 0.09],
        "经济效益": [0.15, 0.20, 0.14, 0.09, 0.15, 0.17],
        "供应能力": [250, 180, 300, 200, 150, 175],
        "市场影响度": [0.23, 0.15, 0.27, 0.30, 0.18, 0.26],
        "交货情况": [0.87, 0.95, 0.99, 0.89, 0.82, 0.94],
    }

    # 创建DataFrame
    df = pd.DataFrame(data, index=[f"方案{i}" for i in range(1, 7)])

    # 定义成本型指标
    cost_indicators = ["产品价格", "地理位置", "售后服务"]

    print("\n测试场景1: 等权重分析")
    print("-" * 30)

    # 测试1: 等权重
    results1 = gra(df, cost_class=cost_indicators, verbose=False)
    print(f"等权重分析结果:")
    print(
        f"最优方案: {results1['ranking'].index[0]} (关联度: {results1['ranking'].iloc[0]:.4f})"
    )

    print("\n测试场景2: 自定义权重分析")
    print("-" * 30)

    # # 测试2: 自定义权重 (重点关注质量、服务和交货)
    # custom_weights = np.array([0.25, 0.10, 0.08, 0.15, 0.12, 0.10, 0.08, 0.08, 0.04])
    # results2 = gra(df, cost_class=cost_indicators, weights=custom_weights, verbose=False)
    # print(f"自定义权重分析结果:")
    # print(f"最优方案: {results2['ranking'].index[0]} (关联度: {results2['ranking'].iloc[0]:.4f})")

    # print("\n测试场景3: 质量导向权重分析")
    # print("-" * 30)

    # # 测试3: 质量导向权重
    # quality_weights = np.array([0.4, 0.05, 0.05, 0.1, 0.15, 0.05, 0.05, 0.05, 0.1])
    # results3 = gra(df, cost_class=cost_indicators, weights=quality_weights, verbose=False)
    # print(f"质量导向权重分析结果:")
    # print(f"最优方案: {results3['ranking'].index[0]} (关联度: {results3['ranking'].iloc[0]:.4f})")

    # # 比较三种权重方案的结果
    # print("\n三种权重方案对比:")
    # print("="*50)
    # comparison_df = pd.DataFrame({
    #     '等权重': results1['ranking'],
    #     '自定义权重': results2['ranking'],
    #     '质量导向': results3['ranking']
    # })
    # print(comparison_df.round(4))

    # # 详细分析最后一个结果
    # print("\n" + "="*60)
    # print("【详细分析 - 质量导向权重方案】")
    # print("="*60)

    # 运行详细分析
    results_detailed = gra(df, cost_class=cost_indicators, verbose=True)

    # 生成可视化图表
    print("\n正在生成可视化图表...")
    plot_gra_results(results_detailed)

    # 生成详细报告
    generate_gra_report(results_detailed)

    print("\n测试完成！")
