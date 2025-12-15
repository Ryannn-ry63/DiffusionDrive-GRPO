import numpy as np
from sklearn.cluster import MiniBatchKMeans

def cluster_trajectories_to_64():
    """
    将8192条轨迹聚类成64条典型轨迹,并转换为(64, 8, 2)格式
    """
    vocab_size_64 = 64
    target_horizon = 8  # 目标时间步
    target_dim = 2      # 目标维度

    # 加载原始轨迹数据
    original_trajectories = np.load('/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/test_8192_kmeans.npy')
    print(f"原始轨迹数据形状: {original_trajectories.shape}")

    # 步骤1：降维 - 从3维降到2维（取前2维，通常是x,y坐标）
    trajectories_2d = original_trajectories[:, :, :2]
    print(f"降维后数据形状: {trajectories_2d.shape}")

    # 步骤2：降时间步 - 从40个时间步降到8个时间步
    # 方法1：均匀采样（推荐）
    original_horizon = trajectories_2d.shape[1]
    indices = np.linspace(0, original_horizon-1, target_horizon, dtype=int)
    trajectories_downsampled = trajectories_2d[:, indices, :]
    
    print(f"降时间步后数据形状: {trajectories_downsampled.shape}")

    # 步骤3：重塑数据以适应K-means输入
    L, HORIZON, DIM = trajectories_downsampled.shape
    all_traj_2d = trajectories_downsampled.reshape(L, -1)
    print(f"重塑后的数据形状: {all_traj_2d.shape}")

    # 步骤4：使用MiniBatchKMeans进行聚类
    clustering = MiniBatchKMeans(
        n_clusters=vocab_size_64,
        batch_size=512,
        verbose=True,
        tol=0.0,
        random_state=42
    ).fit(all_traj_2d)

    # 步骤5：获取聚类中心并重塑回目标格式(64, 8, 2)
    cluster_centers = clustering.cluster_centers_
    typical_trajectories = cluster_centers.reshape(vocab_size_64, target_horizon, target_dim)

    # 保存结果
    save_path = '/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/anchor_64_kmeans.npy'
    np.save(save_path, typical_trajectories)

    print(f"聚类完成！")
    print(f"64条典型轨迹已保存到: {save_path}")
    print(f"输出轨迹形状: {typical_trajectories.shape}")

    return typical_trajectories

if __name__ == '__main__':
    typical_trajs = cluster_trajectories_to_64()