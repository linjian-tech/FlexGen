import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# 绘图参数全家桶
params = {
    'axes.labelsize': '13',
    'xtick.labelsize': '11',
    'ytick.labelsize': '10',
    'legend.fontsize': '10',
    'figure.figsize': '3.5, 2.5',
    'figure.dpi':'300',
    'figure.subplot.left':'0.15',
    'figure.subplot.right':'0.96',
    'figure.subplot.bottom':'0.14',
    'figure.subplot.top':'0.91',
    'pdf.fonttype':'42',
    'ps.fonttype':'42',
}
pylab.rcParams.update(params)

N = 4
memory_size = np.array([1, 30, 300, 3000])*0.336 + 64



ind = np.arange(N)    # the x locations for the groups
width = 0.5       # the width of the bars: can also be len(x) sequence

color_1 = "#F27970"
color_2 = "#BB9727"
color_3 = "#54B345"
color_4 = "#32B897"
color_5 = "#05B9E2"

fig = plt.figure(1)
ax1 = plt.subplot(111)

p1 = plt.bar(ind, memory_size, width, color='none', edgecolor=color_2, hatch="-----", alpha=.99)



plt.ylabel('Memory Cost (TB)')
plt.xlabel('Number of Users')
# plt.legend(loc="upper right")
plt.xticks(ind, ('1', '30', '300', '3000'))

plt.grid(linestyle="--", linewidth=0.5, color='black', alpha = 0.3)

# 设置y轴上限，确保标签可见
y_max = max(memory_size) * 1.2
ax1.set_ylim(0, y_max)

plt.savefig("Multi_User_Memory_Cost.pdf", format = 'pdf', bbox_inches='tight')

plt.show()