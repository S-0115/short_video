import numpy as np
import matplotlib.pyplot as plt

def weibull_cdf(t, beta, eta):
    """Weibull CDF: F(t) = 1 - exp(-(t/eta)**beta)"""
    return 1 - np.exp(-(t / eta) ** beta)

# 给定参数
beta = 1.156
eta  = 7.544          # 举例：β=1.8，η=120

# 取时间点
t = np.linspace(0, 3 * eta, 500)   # 画到 3 倍 eta 足够长

# 计算 CDF
cdf = weibull_cdf(t, beta, eta)

# 绘图
plt.figure(figsize=(5, 3))
plt.plot(t, cdf, label=f'Weibull CDF (β={beta}, η={eta})')
plt.xlabel('t')
plt.ylabel('Cumulative probability F(t)')
plt.title('Weibull CDF')
plt.grid(True)
plt.legend()
plt.ylim(0, 1.05)
plt.show()