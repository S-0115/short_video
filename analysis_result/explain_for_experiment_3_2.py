import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

plt.rcParams.update({
    "font.size": 16,          # 默认文字
    "axes.titlesize": 16,     # 标题
    "axes.labelsize": 16,     # 坐标轴标签
    "xtick.labelsize": 16,    # x 轴刻度
    "ytick.labelsize": 16,    # y 轴刻度
    "legend.fontsize": 16,    # 图例
})

# --- 模拟数据准备 (请替换为你实验中提取的真实数据) ---
# 假设 ct 维度是 64
dim = 20
num_samples = 30

def load_ct(data_path, stime, num_sample):
    time = []
    latent_ct = []
    with open(data_path, 'rb') as f:
        for line in f:
            data_line = line.decode('utf-8').split('#')
            time.append(float(data_line[0]))
            latent_ct.append(np.array(eval(data_line[1].strip())))
    # print(time)
    # print(type(latent_ct[0]))
    start_idx = None
    for i in range(len(time)):
        if time[i] / 1000. >= stime:
            start_idx = i
            break

    latent = []
    for i in range(num_sample):
        latent.append(latent_ct[start_idx+i])
    latent = np.array(latent)

    # print(f"select latent between {stime} to {time[start_idx + num_sample] / 1000.}")
    print(f"select latent between to {time[start_idx + num_sample] / 1000. - 147}")
    return latent

data_path = '../algorithms/secbad/logs/data_for_exp3_2/trace_0_'
# data_path = '../algorithms/secbad/logs/secbad_test/logs_short_video_env/dataset/user_0/trace_0_'
ct_short = load_ct(data_path, stime=150, num_sample=num_samples)
# print(ct_short.shape)

ct_long = load_ct(data_path, stime=220, num_sample=num_samples)
# print(ct_long.shape)

# 合并数据
X = np.vstack([ct_long, ct_short])

# 假设数据同上：X 为 ct 数组 (400, 64), labels 为 ['Long', 'Short']
# y 需要转为数字标签用于 LDA
y = np.array([0] * num_samples + [1] * num_samples)

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
plt.subplots_adjust(bottom=0.4)

# --- 方法一：t-SNE 降维可视化 ---
def plot_tsne(ax, data):
    print("正在进行 t-SNE 降维...")
    tsne = TSNE(n_components=2, perplexity=15, n_iter=1000, random_state=42)
    X_embedded = tsne.fit_transform(data)

    ax.scatter(X_embedded[y==0, 0], X_embedded[y==0, 1], label='Long-view', alpha=0.6)
    ax.scatter(X_embedded[y==1, 0], X_embedded[y==1, 1], label='Short-view', alpha=0.6)

    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    # ax.legend(loc='upper center', frameon=True, fontsize=16, ncols=2, columnspacing=1.2, handlelength=1.6, handletextpad=0.5,  bbox_to_anchor=(0., 1.1, 1., .102))
    ax.set_title('(a) t-SNE', y =-0.4, fontsize=20)
ax1 = axs[0]
# 执行可视化和分析
plot_tsne(ax1, X)

# --- PCA 实现 ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# --- 可视化对比 ---
def plot_pca(ax, data):
    # PCA 子图
    ax.scatter(data[y==0, 0], data[y==0, 1], label='Long-view', alpha=0.6)
    ax.scatter(data[y==1, 0], data[y==1, 1], label='Short-view', alpha=0.6)

    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')

    ax.legend(loc='upper center', frameon=True, fontsize=16, ncols=2, columnspacing=1.5, handlelength=1.5, handletextpad=0.7, bbox_to_anchor=(0., 1.15, 1., .102))
    ax.set_title('(b) PCA', y =-0.4, fontsize=20)
ax2 = axs[1]
plot_pca(ax2, X_pca)

# --- LDA 实现 (注意 LDA 是有监督的，需要 y) ---
lda = LDA(n_components=1) # 对于两分类，LDA 只能降到 1 维，或者投影到特定平面
X_lda = lda.fit_transform(X, y)

def plot_lda(ax, data):
    # PCA 子图
    ax.scatter(X_lda[y==0], np.random.uniform(-0.1, 0.1, size=num_samples), label='Long-view', alpha=0.6)
    ax.scatter(X_lda[y==1], np.random.uniform(-0.1, 0.1, size=num_samples), label='Short-view', alpha=0.6)

    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')

    # ax.legend(loc='upper center', frameon=True, fontsize=16, ncols=2, columnspacing=1.2, handlelength=1.6, handletextpad=0.5,  bbox_to_anchor=(0., 1.1, 1., .102))

    ax.set_title('(c) LDA', y =-0.4, fontsize=20)
ax3 = axs[2]
plot_lda(ax3, X_lda)

# plt.title('t-SNE')
plt.tight_layout(
)
plt.savefig('./explain_for_experiment_3_2.svg')
plt.show()
