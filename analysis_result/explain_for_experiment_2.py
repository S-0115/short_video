import matplotlib.pyplot as plt
import numpy as np

# --- 配置数据 ---
labels = ['1', '2', '3', '4', '5']
# data_ax1 = [916, 106529, 25088, 21266, 21218]
# data_ax2 = [3706, 115710, 126951, 23599, 103586]


data_ax1 = [  553. ,91801. ,30957. ,17747. ,26831.]
data_ax2 = [   930. ,294461.  ,88434.  ,22964.  ,24843.]

# 专业配色方案 (您可以根据喜好更换)
colors = [
    (75 / 255, 116 / 255, 178 / 255),
    (144 / 255, 190 / 255, 224 / 255),
    (230 / 255, 241 / 255, 243 / 255),
    (255 / 255, 223 / 255, 146 / 255),
    (252 / 255, 140 / 255, 90 / 255),
    (219 / 255, 49 / 255, 36 / 255),
]
# # 这里使用 matplotlib 内置的 Blues 渐变色
# cmap = plt.get_cmap("Blues")
# # 选取 0.2 到 0.9 之间的颜色深浅，对应 1-5 程度递增
# colors = cmap(np.linspace(0.2, 0.8, len(labels)))

# --- 创建画布 ---
fig, axs = plt.subplots(1, 2, figsize=(12, 6))


def draw_styled_pie(ax, data, title):
    wedges, texts, autotexts = ax.pie(
        data,
        labels=[''] * 5,
        colors=colors,
        autopct='%1.1f%%',
        pctdistance=1.2,
        startangle=90,
        counterclock=False,
        wedgeprops={'linewidth': 0.5, 'edgecolor': 'black'},
        # textprops={'fontsize': 16, 'fontweight': 'bold', 'color': '#333333'}
        textprops = {'fontsize': 16, 'color': '#333333'}
    )

    ax.legend(
        wedges,
        labels,
        loc="center left",
        bbox_to_anchor=(1.05, 0.25),
        frameon=False,
        title_fontproperties={'weight':'bold'}
    )

    ax.set_xlabel(title, fontsize=16, labelpad=20)


# --- 绘制左图 ---
draw_styled_pie(axs[0], data_ax1, 'Select segment length of [0-12]s')

# --- 绘制右图 ---
draw_styled_pie(axs[1], data_ax2, 'Select segment length of [12-inf]s')

# --- 布局与保存 ---
plt.tight_layout(pad=2.0)
plt.savefig('./explain_for_experiment_2_styled.svg', format='svg', bbox_inches='tight')
plt.show()