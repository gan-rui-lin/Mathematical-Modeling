import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

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

# 蓝紫渐变
custom_blue_purple = LinearSegmentedColormap.from_list(
    "blue_purple", ["#FBF9FA", "#DEDCEA", "#AFB7DB", "#8197C6", "#507AAF"], N=256
)

# 橙红渐变
custom_orange_red = LinearSegmentedColormap.from_list(
    "orange_red", ["#F7F7E9", "#F3E1AF", "#DBBF92", "#D78851", "#BE5C37"], N=256
)

# 绿色系
custom_green = LinearSegmentedColormap.from_list(
    "green", ["#F8FBF6", "#DEEED4", "#BBDCA1", "#86C06C", "#5B9C4B"], N=256
)

# 蓝红渐变

custom_blue_red = LinearSegmentedColormap.from_list(
    "blue_red",
    [
        "#053061",
        "#134b87",
        "#327db7",
        "#6fafd2",
        "#c7e0ed",
        "#fbd2bc",
        "#feab88",
        "#b71c2c",
        "#8b0823",
        "#6a0624",
    ],
)



# ------分界线-----------------

figsize = (10, 7)
big_figsize = (16, 8)

np.random.seed(42)
types = ["高钾", "铅钡", "青铜器", "陶器"]
components = ["SiO2", "CaO", "Fe2O3", "K2O"]
test_data = pd.DataFrame(
    {
        "成分": np.tile(components, 20),
        "变化百分比": np.random.normal(0, 20, 80),
        "文物类型": np.repeat(types, 20),
    }
)

# 画柱状图，以 componets 作为不同的 hue

plt.figure(figsize=figsize)

# colors = plt.cm.Set3(np.linspace(0, 1, len(components)))

ax = sns.barplot(
    data=test_data,
    x="文物类型",
    y="变化百分比",
    hue="成分",
    palette=color_schemes["heavy_1"],
    errorbar=None,
)

for bar in ax.patches:
    # 也就是具体的 y 值
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,  # x 位置：柱子中心
        height
        + (0.01 if height >= 0 else -0.03),  # y 位置：柱顶上方（或下方，若为负值）
        f"{height:.1f}",  # 显示1位小数
        ha="center",
        va="bottom" if height >= 0 else "top",  # 正数在下方，负数在上方
        fontsize=10,
    )

plt.xticks(rotation=45)
plt.axhline(0, color="gray", linestyle="--", alpha=0.7)
plt.legend(title="成分", loc="upper right")
plt.tight_layout()
plt.savefig("barplot.png", dpi=300)
plt.show()



# 画箱线图

plt.figure(figsize=figsize)

sns.boxplot(
    data=test_data,
    x="文物类型",
    y="变化百分比",
    hue="成分",
    palette=color_schemes["heavy_1"],
)

plt.xticks(rotation=45)
plt.legend(title="成分", loc="upper right")
plt.tight_layout()
plt.savefig("boxplot.png", dpi=300)
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=big_figsize)

sns.barplot(
    data=test_data,
    x="文物类型",
    y="变化百分比",
    hue="成分",
    palette=color_schemes["heavy_1"],
    ax=ax1,
    errorbar=None,
)

ax1.tick_params(axis="x", rotation=45)
ax1.axhline(0, color="gray", linestyle="--", alpha=0.7)
ax1.legend(title="成分", loc="upper right")

sns.boxplot(
    data=test_data,
    x="文物类型",
    y="变化百分比",
    hue="成分",
    palette=color_schemes["heavy_1"],
    ax=ax2,
)

ax2.tick_params(axis="x", rotation=45)
ax2.legend(title="成分", loc="upper right")

plt.tight_layout()
plt.savefig("bar_boxplot.png", dpi=300)
plt.show()




# 画散点图

# 近似正弦函数 + 随机扰动

x = np.linspace(0, 10, 1000)
y = np.sin(x) + np.random.normal(0, 0.1, 1000)

plt.figure(figsize=figsize)
sns.scatterplot(x=x, y=y, alpha=0.7, marker=".", s=30, lw=0, palette="deep")
plt.xlabel("X轴")
plt.ylabel("Y轴")
plt.tight_layout()
plt.savefig("scatter_plot.png", dpi=300)
plt.show()


zhibiao = ["指标A", "指标B", "指标C", "指标D", "指标E"]
score = [[80, 90, 70, 85, 75], [75, 85, 100, 80, 70]]

df = pd.DataFrame(score, columns=zhibiao, index=["样本1", "样本2"])
# 画雷达图

N = len(list(df.columns))

angles = [n / float(N) * 2 * np.pi for n in range(N)]

angles += angles[:1]

plt.figure(figsize=figsize)

ax = plt.subplot(111, polar=True)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 不重复给 0 角度位置打 x 轴标签
plt.xticks(angles[:-1], df.columns.tolist(), color="black", size=12)

ax.tick_params(axis="x", rotation=0)

ax.set_rlabel_position(0)

plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="black", size=10)

plt.ylim(0, 100)

for i, row in enumerate(df.iterrows()):
    # print(row[1])
    values = row[1].tolist()
    values += values[:1]
    # print(values)
    color = color_schemes["light_1"][i % len(color_schemes["light_1"])]
    ax.plot(angles, values, color=color, linewidth=1, linestyle="solid", label=row[0])
    ax.fill(angles, values, color=color, alpha=0.5)

plt.savefig("radar_chart.png", dpi=300)

plt.show()
