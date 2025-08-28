import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

# 颜色配置
heavy_blue: str = "#5A9AD4"
heavy_purple: str = "#8A8AE4"
heavy_green: str = "#72AF47"
heavy_orange: str = "#FF9655"
heavy_red: str = "#EA6B66"
heavy_gray: str = "#A8A8A8"

light_blue: str = "#B8D8F8"
light_purple: str = "#D4D4FF"
light_green: str = "#C8E6A8"
light_orange: str = "#FFD8A8"
light_red: str = "#FFB6B3"
light_gray: str = "#E0E0E0"

styles_count: int = 5

color_schemes: dict[str, list[str]] = {
    "heavy_1": [heavy_blue, heavy_purple, heavy_green, heavy_orange, heavy_red],
    "light_1": [light_blue, light_purple, light_green, light_red, light_orange],
}


def auto_barplot_ax(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    ax,
    hue_col: Optional[str] = None,
    rotation: int = 45,
    title: str = "",
    palette=None,
) -> None:
    """
    支持传入ax的自动配色柱状图
    """
    if palette is None:
        n_group = data[hue_col].nunique() if hue_col else 1
        palette = (
            color_schemes["light_1"]
            if n_group <= styles_count
            else ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        )

    if hue_col:
        # 分组柱状图
        unique_hues = data[hue_col].unique()
        x_unique = data[x_col].unique()
        # 提供绘制的位置
        x_pos = np.arange(len(x_unique))
        width = 0.8 / len(unique_hues)

        for i, hue in enumerate(unique_hues):
            subset = data[data[hue_col] == hue]
            values = [
                (
                    subset[subset[x_col] == x][y_col].iloc[0]
                    if len(subset[subset[x_col] == x]) > 0
                    else 0
                )
                for x in x_unique
            ]
            color = palette[i % len(palette)] if isinstance(palette, list) else None
            ax.bar(
                x_pos + i * width - (len(unique_hues) - 1) * width / 2,
                values,
                width,
                label=hue,
                alpha=0.8,
                color=color,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_unique, rotation=rotation)
        ax.legend()
    else:
        # 简单柱状图
        x_data = data[x_col]
        y_data = data[y_col]
        colors = palette[: len(x_data)] if isinstance(palette, list) else palette
        ax.bar(x_data, y_data, alpha=0.8, color=colors)
        if rotation != 0:
            ax.tick_params(axis="x", rotation=rotation)

    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_vikor_results(
    results_df: pd.DataFrame,
    normalized_data: pd.DataFrame,
    S_values: np.ndarray,
    R_values: np.ndarray,
    v: float,
) -> None:
    """
    绘制VIKOR结果的可视化图表
    """
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]  # 解决中文显示
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. S值、R值、Q值对比 - 使用自定义颜色
    metrics_data = []
    for _, row in results_df.iterrows():
        metrics_data.extend(
            [
                {
                    "方案": row["方案"],
                    "指标": "S值(群体效用)",
                    "数值": row["S值(群体效用)"],
                },
                {
                    "方案": row["方案"],
                    "指标": "R值(个体遗憾)",
                    "数值": row["R值(个体遗憾)"],
                },
                {
                    "方案": row["方案"],
                    "指标": "Q值(VIKOR指数)",
                    "数值": row["Q值(VIKOR指数)"],
                },
            ]
        )

    metrics_df = pd.DataFrame(metrics_data)
    auto_barplot_ax(
        data=metrics_df,
        x_col="方案",
        y_col="数值",
        hue_col="指标",
        ax=ax1,
        title="VIKOR方法各指标对比",
        palette=color_schemes["heavy_1"],
    )
    ax1.set_ylabel("数值")

    # 2. 权重敏感性分析
    v_range = np.linspace(0, 1, 11)
    Q_sensitivity = np.zeros((len(normalized_data), len(v_range)))

    # 重新计算S和R的范围
    S_star, S_minus = S_values.min(), S_values.max()
    R_star, R_minus = R_values.min(), R_values.max()

    for j, v_test in enumerate(v_range):
        for i in range(len(normalized_data)):
            if S_minus != S_star and R_minus != R_star:
                Q_test = v_test * (S_values[i] - S_star) / (S_minus - S_star) + (
                    1 - v_test
                ) * (R_values[i] - R_star) / (R_minus - R_star)
            else:
                Q_test = S_values[i] if S_minus != S_star else R_values[i]
            Q_sensitivity[i, j] = Q_test

    colors = color_schemes["heavy_1"]
    for i, aircraft in enumerate(normalized_data.index):
        color = colors[i % len(colors)]
        ax2.plot(
            v_range, Q_sensitivity[i], "o-", label=aircraft, color=color, linewidth=2
        )

    ax2.set_xlabel("决策机制系数 v")
    ax2.set_ylabel("Q值")
    ax2.set_title("Q值对权重系数v的敏感性分析")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvline(
        x=v, color=heavy_red, linestyle="--", alpha=0.7, linewidth=2, label=f"当前v={v}"
    )

    plt.tight_layout()
    plt.show()


def vikor(
    data: pd.DataFrame, weights: np.ndarray, cost_class=None, v=0.5, verbose=True
):
    """
    VIKOR方法实现
    data: pd.DataFrame, 行为方案，列为指标
    weights: np.ndarray, 指标权重（和为1）
    cost_class: list, 极小型指标名称
    v: float, 决策机制系数
    verbose: bool, 是否打印详细过程
    返回: 排名结果DataFrame
    """
    if cost_class is None:
        cost_class = []

    # 步骤1: 极小型指标转极大型并归一化
    normalized_data = data.copy()
    for col in data.columns:
        col_max = data[col].max()
        if col in cost_class:
            normalized_data[col] = 1 - data[col] / col_max
        else:
            normalized_data[col] = data[col] / col_max

    # 步骤2: 计算f*和f-
    f_star = normalized_data.max()
    f_minus = normalized_data.min()

    # 步骤3: 计算S值和R值
    S_values = []
    R_values = []
    for i, scheme in enumerate(normalized_data.index):
        diff = f_star - normalized_data.loc[scheme]
        range_val = f_star - f_minus
        normalized_diff = np.where(range_val > 0, diff / range_val, 0)
        S_i = np.sum(weights * normalized_diff)
        R_i = np.max(weights * normalized_diff)
        S_values.append(S_i)
        R_values.append(R_i)
    S_values = np.array(S_values)
    R_values = np.array(R_values)
    S_star, S_minus = S_values.min(), S_values.max()
    R_star, R_minus = R_values.min(), R_values.max()

    # 步骤4: 计算Q值
    Q_values = []
    for i in range(len(normalized_data)):
        if S_minus != S_star and R_minus != R_star:
            Q_i = v * (S_values[i] - S_star) / (S_minus - S_star) + (1 - v) * (
                R_values[i] - R_star
            ) / (R_minus - R_star)
        else:
            Q_i = (
                v * (S_values[i] - S_star)
                if S_minus != S_star
                else (1 - v) * (R_values[i] - R_star) if R_minus != R_star else 0
            )
        Q_values.append(Q_i)
    Q_values = np.array(Q_values)

    # 步骤5: 汇总结果
    results_df = pd.DataFrame(
        {
            "方案": normalized_data.index,
            "S值(群体效用)": S_values,
            "R值(个体遗憾)": R_values,
            "Q值(VIKOR指数)": Q_values,
        }
    )
    results_df = results_df.sort_values("Q值(VIKOR指数)")
    results_df["排名"] = range(1, len(results_df) + 1)

    # # 步骤6: 条件检验
    # n = len(results_df)
    # best_alternative = results_df.iloc[0]['方案']
    # second_best = results_df.iloc[1]['方案']
    # Q_best = results_df.iloc[0]['Q值(VIKOR指数)']
    # Q_second = results_df.iloc[1]['Q值(VIKOR指数)']
    # DQ = 1 / (n - 1)
    # advantage_condition = (Q_second - Q_best) >= DQ
    # S_ranking = np.argsort(S_values)
    # R_ranking = np.argsort(R_values)
    # best_idx = list(normalized_data.index).index(best_alternative)
    # stability_condition = (S_ranking[0] == best_idx) or (R_ranking[0] == best_idx)

    if verbose:
        # print(results_df.round(4))
        # print(f"\n条件1 - 可接受优势: {'满足' if advantage_condition else '不满足'}")
        # print(f"条件2 - 可接受稳定性: {'满足' if stability_condition else '不满足'}")
        # if advantage_condition and stability_condition:
        #     print(f"✅ 推荐方案: {best_alternative}")
        # elif advantage_condition:
        #     print(f"⚠️ 推荐方案: {best_alternative}（稳定性有待验证）")
        # else:
        #     compromise_set = []
        #     for i in range(len(results_df)):
        #         if results_df.iloc[i]['Q值(VIKOR指数)'] - Q_best < DQ:
        #             compromise_set.append(results_df.iloc[i]['方案'])
        #     print(f"⚠️ 需要考虑妥协解集: {compromise_set}")

        # 绘制可视化图表
        plot_vikor_results(results_df, normalized_data, S_values, R_values, v)


if __name__ == "__main__":
    data = [
        # [最大速度, 飞行半径, 最大负载, 费用, 可靠性, 灵敏度]
        [2.0, 1500, 20000, 5500000, 0.5, 1.0],  # A1
        [2.5, 2700, 18000, 6500000, 0.3, 0.5],  # A2
        [1.8, 2000, 21000, 4500000, 0.7, 0.7],  # A3
        [2.2, 1800, 20000, 5000000, 0.5, 0.5],  # A4
    ]

    aircraft_names = ["A1", "A2", "A3", "A4"]
    indicator_names = ["最大速度", "飞行半径", "最大负载", "费用", "可靠性", "灵敏度"]

    df = pd.DataFrame(data=data, columns=indicator_names, index=aircraft_names)

    weights = np.array([1 / 6] * 6)  # 等权重
    cost_class = ["费用"]

    print("VIKOR方法示例：")
    vikor(df, weights, cost_class=cost_class, v=0.5, verbose=True)
