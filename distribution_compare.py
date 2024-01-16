
import matplotlib.pyplot as plt
import numpy as np
cifar_distribute = np.bincount(np.load("/public/home/renhui/code/Imbalanced_clustering/PPOT_dev/results/CIAR100_0.01.npy"))
ppot_distribute = np.bincount(np.load("/public/home/renhui/code/Imbalanced_clustering/PPOT_dev/results/ours_distribution.npy"))
iic_distribute = np.bincount(np.load("/public/home/renhui/code/Imbalanced_clustering/PPOT_dev/results/iic_distribution.npy"))
selflabel_distribute = np.bincount(np.load("/public/home/renhui/code/Imbalanced_clustering/PPOT_dev/results/scan*_distribution.npy"))
sppot_distribute = np.bincount(np.load("/public/home/renhui/code/Imbalanced_clustering/PPOT_structure/results/sppot_CIFAR100_0.01.npy"))

plt.subplots(1,5, figsize=(25,5), sharey=True)


plt.subplot(1,5,1)
plt.bar(range(len(cifar_distribute)), cifar_distribute, color = "dodgerblue")
# plt.xlabel("Class index", fontsize = 14)
plt.ylabel("Num of images", fontsize = 12)
plt.title("Training set distribution")
plt.xlim([0,100])

plt.subplot(1,5,2)
plt.bar(range(len(selflabel_distribute)), selflabel_distribute, color = "dodgerblue")
# plt.xlabel("Class index", fontsize = 14)
plt.title("SCAN* prediction distribution")
plt.xlim([0,100])

plt.subplot(1,5,3)
plt.bar(range(len(iic_distribute)), iic_distribute, color = "dodgerblue")
plt.xlabel("Class index", fontsize = 14)
plt.title("IIC prediction distribution")
plt.xlim([0,100])
plt.ylabel("Num of images", fontsize = 12)

plt.subplot(1,5,4)
plt.bar(range(len(ppot_distribute)), ppot_distribute, color = "dodgerblue")
plt.xlabel("Class index", fontsize = 14)
plt.title("P$^2$OT prediction distribution")
plt.xlim([0,100])

plt.subplot(1,5,5)
plt.bar(range(len(sppot_distribute)), sppot_distribute, color = "dodgerblue")
plt.xlabel("Class index", fontsize = 14)
plt.title("SP$^2$OT prediction distribution")
plt.xlim([0,100])

plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)


plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.tight_layout()
plt.savefig(f"./distribution_compare.pdf", dpi=300)
plt.show()

def plot_distribution(distribution,dataset_name,train=True):
    plt.bar(range(len(distribution)), distribution, color = "dodgerblue")
    plt.title(f"{dataset_name}", fontsize = 14)
    plt.xlabel("Class index", fontsize = 14)
    plt.ylabel("Num of images", fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    # mean = sum(distribution) / len(distribution)
    # plt.plot([0, len(distribution)], [mean, mean], color='red', linestyle='--')
    # plt.legend(["mean"])
    # plt.ylim([0,1000])
    plt.savefig(f"./{dataset_name} {'trainset' if train else 'valset'} class_distribution.pdf")
    plt.show()
    plt.close()
    head_num = int(len(distribution) * 0.3)
    medium_num = int(len(distribution) * 0.4)
    tail_num = len(distribution) - head_num - medium_num
    head = distribution[:head_num]
    medium = distribution[head_num:head_num + medium_num]
    tail = distribution[-tail_num:]
    print(dataset_name)
    print(f"head:{head_num},medium:{medium_num},tail:{tail_num}")
    medium_mean = sum(medium) / len(medium)
    tail_mean = sum(tail) / len(tail)
    print(f"medium_mean:{medium_mean},tail_mean:{tail_mean}")
    print(f"tail_mean/medium_mean:{tail_mean/medium_mean}")