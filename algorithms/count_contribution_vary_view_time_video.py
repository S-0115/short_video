import matplotlib.pyplot as plt


algorithm = 'dashlet'
file_path = algorithm + '/logs/dataset/user_0/trace_for_exp3_14'
chunk_num = []
chunk_all = []
qoe_avg = []

with open(file_path, 'r') as f:
    for line in f:
        # print(line)
        infos = line.split(',')
        chunk_num.append(int(infos[0]))
        chunk_all.append(int(infos[1]))
        qoe_avg.append(float(infos[2]) / float(infos[0]))
view_percentage = []
for i in range(len(qoe_avg)):
    view_percentage.append(chunk_num[i] / chunk_all[i])

plt.figure(1,(12,6))
plt.scatter(view_percentage, qoe_avg)
plt.xlabel('View Percentage')
plt.xlim(0, 1)
plt.ylabel('QoE Average')
# plt.ylim(-20, 25)
plt.show()