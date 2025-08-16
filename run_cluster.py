import numpy as np
import os
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax


def dtw_kmeans(X, n_clusters=10, max_iter=10, random_state=42, verbose=True):
    # 先对所有时间序列进行归一化处理
    X_scaled = TimeSeriesScalerMinMax().fit_transform(X)
    cpu_count = os.cpu_count()
    max_workers = max(1, cpu_count // 4)
    # 使用 tslearn 的 DTW 距离版本的 KMeans
    model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="dtw",
        max_iter=max_iter,
        verbose=verbose,
        random_state=random_state,
        n_jobs=max_workers
    )
    labels = model.fit_predict(X_scaled)
    return labels


def get_each_cluster_group_idx(n_clusters, start_date, end_date):
    dir_name = 'S' + (start_date + '_E' + end_date).replace('-', '')

    all_address = os.listdir(f'./datasets/financial/{dir_name}')
    print(len(all_address))

    min_length = 1e9
    # for file in all_address:
        # with open(os.path.join(f'./datasets/financial/{dir_name}', file), 'rb') as f:
            # now_df = pickle.load(f)
            # min_length = min(min_length, len(now_df))

    # 这里假设所有数据的长度都大于等于 776，实际使用时可以根据需要调整
    min_length = 827  
    train_length = int(min_length * 0.7)

    X = []
    all_func_code_list = []
    for file in all_address:
        with open(os.path.join(f'./datasets/financial/{dir_name}', file), 'rb') as f:
            now_df = pickle.load(f)
        if len(now_df) < min_length:
            continue
        selected_seq = now_df[-min_length:, :]
        selected_seq = selected_seq[:train_length, :]
        X.append(selected_seq[:, -1])  # 只取特征部分
        all_func_code_list.append(selected_seq[:, 0])  # 保存 func_code

    X = np.array(X, dtype=object)  # list of (train_length, 3)
    all_func_code = np.stack(all_func_code_list)
    print("X shape:", len(X), "with each of shape:", X[0].shape)

    labels = dtw_kmeans(X, n_clusters=n_clusters, max_iter=10, verbose=True)
    print("聚类标签:", labels)

    representative_func_codes = all_func_code[:, 0]
    mapping_array = np.column_stack((representative_func_codes, labels))

    os.makedirs('./datasets', exist_ok=True)
    with open(f'./datasets/func_code_to_label_{n_clusters}.pkl', 'wb') as f:
        pickle.dump(mapping_array, f)

    print(f"映射表已保存至 './datasets/func_code_to_label_{n_clusters}.pkl'")

    return mapping_array


def plot_clusters_from_mapping(mapping_array, data_dir, output_dir='./figs/clusters'):
    """
    根据 mapping_array 绘制每个 fund 的聚类图并保存

    参数：
    - mapping_array: 每行包含 [fund_code, cluster_label, filename]
    - data_dir: 存放原始数据的文件夹路径
    - output_dir: 图像保存路径
    """
    all_address = os.listdir(f'./datasets/financial/{dir_name}')
    min_length = 1e9
    for file in all_address:
        with open(os.path.join(f'./datasets/financial/{dir_name}', file), 'rb') as f:
            now_df = pickle.load(f)
            min_length = min(min_length, len(now_df))

    train_length = int(min_length * 0.7)

    for fund_code, label in tqdm(mapping_array):
        file_path = os.path.join(data_dir, fund_code + '.pkl')
        with open(file_path, 'rb') as f:
            now_df = pickle.load(f)

        # 重新对数据截取相同片段（注意需与聚类时一致）
        selected_seq = now_df[-min_length:, :]
        selected_seq = selected_seq[:train_length, :]
        seq = selected_seq[:, -1]  # 特征值

        # 保存图像
        group_dir = os.path.join(output_dir, str(int(label)))
        os.makedirs(group_dir, exist_ok=True)

        plt.figure(figsize=(10, 4))
        plt.plot(seq, label=f'Fund {fund_code}')
        plt.title(f'Cluster {label} - Fund {fund_code}')
        plt.xlabel('Time Step')
        plt.ylabel('Feature Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(group_dir, f'{fund_code}.png'))
        plt.close()


def balanced_cluster(data, cluster_num, group_size_limit=100):
    import random
    import pickle
    import numpy as np
    from collections import defaultdict

    arr = np.array(data)
    samples = arr.tolist()

    # 构建原始组 → 样本映射
    group_to_samples = defaultdict(list)
    for fund_code, group in samples:
        group_to_samples[group].append(fund_code)

    # 累积所有切好的 batch（每个组大小不超过 group_size_limit）
    all_batches = []
    tail_samples = []  # 存储无法完全达到 group_size_limit 的样本

    for fund_codes in group_to_samples.values():
        random.shuffle(fund_codes)
        for i in range(0, len(fund_codes), group_size_limit):
            batch = fund_codes[i:i + group_size_limit]
            if len(batch) < group_size_limit:
                tail_samples.append(batch)  # 单独保留的小组
            else:
                all_batches.append(batch)

    # 如果有剩余样本单独作为一个新的组
    if tail_samples:
        # 每个残余的组单独做为一个新的 group
        for batch in tail_samples:
            all_batches.append(batch)

    # 重新分配连续组号
    new_samples = []
    for new_group_id, batch in enumerate(all_batches):
        for fund_code in batch:
            new_samples.append([fund_code, int(new_group_id)])

    new_arr = np.array(new_samples)

    # 保存
    with open(f'./datasets/func_code_to_label_{cluster_num}_balanced.pkl', 'wb') as f:
        pickle.dump(new_arr.tolist(), f)

    print(f"✅ 新数据保存完毕，组数: {len(all_batches)}，总样本数: {len(new_samples)}，shape: {new_arr.shape}")
    return new_arr



if __name__ == '__main__':
    from utils.exp_config import get_config
    config = get_config('FinancialConfig')

    n_clusters = config.n_clusters
    # mapping_array = get_each_cluster_group_idx(n_clusters, config.start_date, config.end_date)
    with open(f'./datasets/func_code_to_label_{n_clusters}.pkl', 'rb') as f:
        mapping_array = pickle.load(f)

    # 构造数据目录名（必须和聚类用的保持一致）
    dir_name = 'S' + (config.start_date + '_E' + config.end_date).replace('-', '')
    data_dir = f'./datasets/financial/{dir_name}'
    output_dir = f'./figs/clusters_{n_clusters}'
    # 绘图
    # plot_clusters_from_mapping(mapping_array, data_dir, output_dir)

    mapping_array = balanced_cluster(mapping_array, n_clusters)
