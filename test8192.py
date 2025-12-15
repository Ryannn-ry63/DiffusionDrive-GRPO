import numpy as np
from sklearn.cluster import MiniBatchKMeans


def cluster_trajectories_to_64():
    """
    将8192条轨迹聚类成64条典型轨迹
    """
    # 1. 加载现有的8192条轨迹数据
    vocab_size_64 = 64

    # 加载原始轨迹数据
    original_trajectories = np.load('/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/test_8192_kmeans.npy')

    print(f"原始轨迹数据形状: {original_trajectories.shape}")

    # 2. 重塑数据以适应K-means输入
    L, HORIZON, DIM = original_trajectories.shape
    all_traj_2d = original_trajectories.reshape(L, -1)

    print(f"重塑后的数据形状: {all_traj_2d.shape}")

    # 3. 使用MiniBatchKMeans进行聚类
    clustering = MiniBatchKMeans(
        n_clusters=vocab_size_64,
        batch_size=512,
        verbose=True,
        tol=0.0,
        random_state=42
    ).fit(all_traj_2d)

    # 4. 获取聚类中心并重塑回轨迹格式
    cluster_centers = clustering.cluster_centers_
    typical_trajectories = cluster_centers.reshape(vocab_size_64, HORIZON, DIM)

    # 5. 保存结果到指定路径
    save_path = '/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/test_64_kmeans.npy'
    np.save(save_path, typical_trajectories)

    print(f"聚类完成！")
    print(f"64条典型轨迹已保存到: {save_path}")
    print(f"输出轨迹形状: {typical_trajectories.shape}")

    return typical_trajectories


if __name__ == '__main__':
    typical_trajs = cluster_trajectories_to_64()