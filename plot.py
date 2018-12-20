import matplotlib.pyplot as plt
import pickle
import numpy as np
import statistics


file_cloud = pickle.load(open('rewards_intervene_cloud.p', 'rb'))
file_edge = pickle.load(open('rewards_intervene_edge.p', 'rb'))

file_cloud_kd = pickle.load(open('rewards_intervene_cloud_kd.p', 'rb'))
file_edge_kd = pickle.load(open('rewards_intervene_edge_kd.p', 'rb'))

file_cloud_kd_wm = pickle.load(open('rewards_intervene_cloud_wm_kd.p', 'rb'))
file_edge_kd_wm = pickle.load(open('rewards_intervene_edge_wm_kd.p', 'rb'))

file_cloud_dr = pickle.load(open('rewards_intervene_cloud_dr.p', 'rb'))
file_edge_dr = pickle.load(open('rewards_intervene_edge_dr.p', 'rb'))

file_drone = pickle.load(open('rewards_no_intervene.p', 'rb'))


print(sum(file_edge)/ len(file_edge), sum(file_drone)/ len(file_drone), sum(file_cloud)/len(file_cloud))
print(statistics.stdev(file_edge), statistics.stdev(file_drone), statistics.stdev(file_cloud))



fig, ax = plt.subplots()


#axs[0].boxplot(file_drone, 0, '')
#axs[0].set_title('Drone')

#axs[1].boxplot(file_edge, 0, '')
#axs[1].set_title('Edge')

#axs[2].boxplot(file_cloud, 0, '')
#axs[2].set_title('Cloud')

# bp = ax .boxplot([file_drone, file_edge, file_cloud], sym='', positions=[1, 2, 3],
#                     notch=2, bootstrap=1000)


# ax.set_ylabel('Landing Reward')
# ax.set_xlabel('Drone\'s Decision System')
# ax.set_xticklabels(['Drone', 'Edge', 'Cloud'],
#                     rotation=0, fontsize=12)

# cloud_success = len(np.where(np.array(file_cloud) > 0)[0])/ len(file_cloud)
# edge_success = len(np.where(np.array(file_edge) > 0)[0])/ len(file_edge)
# drone_success = len(np.where(np.array(file_drone) > 0)[0])/ len(file_drone)
#
# cloud_success_kd = len(np.where(np.array(file_cloud_kd) > 0)[0])/ len(file_cloud_kd)
# edge_success_kd = len(np.where(np.array(file_edge_kd) > 0)[0])/ len(file_edge_kd)
#
# cloud_success_dr = len(np.where(np.array(file_cloud_dr) > 0)[0])/ len(file_cloud_dr)
# edge_success_dr = len(np.where(np.array(file_edge_dr) > 0)[0])/ len(file_edge_dr)
#
# p1 = plt.bar([1.12], [drone_success], 0.2)
# p2 = plt.bar([1.9, 2.9], [edge_success_kd, cloud_success_kd], 0.2)
# p3 = plt.bar([2.12, 3.12], [edge_success_dr, cloud_success_dr], 0.2)
# p4 = plt.bar([2.34, 3.34], [edge_success, cloud_success], 0.2)
# plt.xticks([1.12, 2.12, 3.12], ('Drone', 'Edge', 'Cloud'))
#
# plt.ylabel('Successful landing probability')
# plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Normal action', 'Prev. action', 'Rand. action', 'No action'))

#plt.plot(range(len(file_cloud)), file_cloud)
#plt.plot(range(len(file_edge)), file_edge)
#plt.plot(range(len(file_drone)), file_drone)

#plt.legend(['Drone', 'Edge', 'Cloud'], fontsize=12, loc="lower right")
#plt.xlabel('samples', fontsize=14)
#plt.ylabel('landing reward', fontsize=14)

# bp = ax .boxplot([file_edge_kd, file_edge_kd_wm, file_cloud_kd, file_cloud_kd_wm], sym='', positions=[1, 2, 3, 4],
#                     notch=1, bootstrap=1000, patch_artist=True)
# ax.set_ylabel('Landing Reward')
# ax.set_xticklabels(['Edge', 'Edge', 'Cloud', 'Cloud'],
#                      rotation=0, fontsize=12)
# colors = ['lightblue', 'pink', 'lightblue', 'pink']
# for patch, color in zip(bp['boxes'], colors):
#         patch.set_facecolor(color)
#
# plt.legend([bp['boxes'][0], bp['boxes'][1]],['Without mobility', 'With mobility'], fontsize=12, loc="lower left")


network_bandwidth = [10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05]
edge_list_reward = [73.9770, 73.30340, 66.07020, 63.59095, 55.91535, 45.05042, -13.14956, -155.91419]
cloud_list_reward = [62.36873, 56.48333, 50.78344, 36.91023, 32.86447, 4.82294, -148.40333, -157.93584]

p1 = plt.plot(network_bandwidth, edge_list_reward, marker='.', label='Edge')
p2 = plt.plot(network_bandwidth, cloud_list_reward, marker='.', label='Cloud')
plt.xlabel('Bandwidth (Mbps)')
plt.ylabel('Landing reward')
plt.legend(fontsize=12)
#plt.show()
plt.savefig('../edge_cloud_network_effect.pdf')