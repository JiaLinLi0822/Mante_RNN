from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from scipy.linalg import qr

from env import *
from net import *

def test_model(model, num_trials=1000, T=20, device='cpu'):
    model.eval()
    correct = 0
    env = RDM()
    with torch.no_grad():
        for _ in range(num_trials):
            input_seq, targets, _, _, _ = env.generate_trial()
            output = model(input_seq)
            pred = 1.0 if output.item() >= 0 else -1.0
            if pred == targets.item():
                correct += 1
    accuracy = correct / num_trials
    print(f"Test Accuracy over {num_trials} trials: {accuracy*100:.2f}%")
    return accuracy

# ===============================
# Record state trajectories and trial metadata
# ===============================
def record_trajectories(model, num_trials=50, T=20, device='cpu'):
    """
    Run multiple trials to record the network's hidden state trajectories.
    Returns:
       - all_states: a numpy array of shape [num_trials, T, hidden_size]
       - trial_info: a list of dictionaries for each trial, each containing
           'motion_coherences', 'color_coherences', and 'context_flags'
         Note: context_flags are now assumed to be 0 and 1.
    """
    model.eval()
    all_states = []
    trial_info = []
    env = RDM()  # Assume RDM environment is defined elsewhere
    with torch.no_grad():
        for i in range(num_trials):
            input_seq, target, motion_coherences, color_coherences, context_flags = env.generate_trial()
            # Get hidden states for each timestep, shape: [T, 1, hidden_size]
            states, output = model.forward_with_states(input_seq)
            # Remove batch dimension -> [T, hidden_size]
            states = states.squeeze(1)
            all_states.append(states.cpu().numpy())
            trial_info.append({
                'motion_coherences': motion_coherences,
                'color_coherences': color_coherences,
                'context_flags': context_flags,  # context_flags are 0 (color task) or 1 (motion task)
                'target': target,
            })
    all_states = np.array(all_states)  # shape: [num_trials, T, hidden_size]
    return all_states, trial_info


# ===============================
# Perform PCA on state trajectories and plot 2D and 3D projections by grouping trials
# ===============================
# def perform_pca_and_plot(state_trajectories, trial_info):
#     """
#     Perform PCA on the recorded hidden state trajectories and plot both 2D and 3D projections.
#     Two sets of figures will be produced:
#        - For the motion task (context_flags == 1): merge over color_coherences and group by motion_coherences.
#        - For the color task (context_flags == 0): merge over motion_coherences and group by color_coherences.
       
#     Parameters:
#        state_trajectories: numpy array of shape [num_trials, T, hidden_size]
#        trial_info: list of dictionaries, each containing 'motion_coherences', 'color_coherences', and 'context_flags'
#     """
#     num_trials, T, hidden_size = state_trajectories.shape
#     # Create a DataFrame for trial metadata and add a trial index
#     trial_df = pd.DataFrame(trial_info)
#     trial_df['trial'] = np.arange(num_trials)
#     trial_df['motion_coherences'] = trial_df['motion_coherences'].apply(lambda x: x.item() if isinstance(x, np.ndarray) and x.size==1 else x)
#     trial_df['color_coherences'] = trial_df['color_coherences'].apply(lambda x: x.item() if isinstance(x, np.ndarray) and x.size==1 else x)
    
#     # ---------------------------
#     # Motion Task (context_flags == 0)
#     # ---------------------------
#     motion_trials = trial_df[trial_df['context_flags'] == 0]
#     if not motion_trials.empty:
#         indices = motion_trials['trial'].values
#         states_motion = state_trajectories[indices]  # shape: [n_trials_motion, T, hidden_size]
#         # Flatten all states for PCA
#         all_states_motion = states_motion.reshape(-1, hidden_size)
#         pca_motion = PCA(n_components=3)
#         pcs_motion = pca_motion.fit_transform(all_states_motion)
#         explained_variance_motion = pca_motion.explained_variance_ratio_
#         print("Motion task: Explained variance ratio of PC1, PC2, PC3:", explained_variance_motion)
#         # Reshape PCA result back to [n_trials_motion, T, 3]
#         pcs_motion_reshaped = pcs_motion.reshape(len(indices), T, 3)
#         motion_trials = motion_trials.copy()
#         motion_trials['pcs'] = list(pcs_motion_reshaped)
        
#         # Group by motion_coherences and compute the average trajectory (averaged over trials)
#         unique_motion = sorted(motion_trials['motion_coherences'].unique())
#         print("Unique motion coherences:", unique_motion)
        
#         # ----- 2D Projection (PC1 vs PC2) -----
#         plt.figure(figsize=(8, 6))
#         for coherence in unique_motion:
#             group_trials = motion_trials[motion_trials['motion_coherences'] == coherence]
#             trajs = np.stack(group_trials['pcs'].values, axis=0)  # shape: [n_trials_in_group, T, 3]
#             avg_traj = trajs.mean(axis=0)  # shape: [T, 3]
#             plt.plot(avg_traj[:, 0], avg_traj[:, 1], marker='o', label=f"Motion coherence: {coherence}")
#         plt.xlabel("PC1")
#         plt.ylabel("PC2")
#         plt.title("PCA Trajectories for Motion Task (context_flags==1)\n(Merged over color_coherences) - 2D")
#         plt.legend()
#         plt.show()
        
#         # ----- 3D Projection -----
#         from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection='3d')
#         for coherence in unique_motion:
#             group_trials = motion_trials[motion_trials['motion_coherences'] == coherence]
#             trajs = np.stack(group_trials['pcs'].values, axis=0)
#             avg_traj = trajs.mean(axis=0)
#             ax.plot(avg_traj[:, 0], avg_traj[:, 1], avg_traj[:, 2], marker='o', label=f"Motion coherence: {coherence}")
#         ax.set_xlabel("PC1")
#         ax.set_ylabel("PC2")
#         ax.set_zlabel("PC3")
#         ax.set_title("PCA Trajectories for Motion Task (context_flags==1)\n(Merged over color_coherences) - 3D")
#         ax.legend()
#         plt.show()
#     else:
#         print("No trials found for Motion Task (context_flags==1).")
    
#     # ---------------------------
#     # Color Task (context_flags == 0)
#     # ---------------------------
#     color_trials = trial_df[trial_df['context_flags'] == 1]
#     if not color_trials.empty:
#         indices = color_trials['trial'].values
#         states_color = state_trajectories[indices]  # shape: [n_trials_color, T, hidden_size]
#         all_states_color = states_color.reshape(-1, hidden_size)
#         pca_color = PCA(n_components=3)
#         pcs_color = pca_color.fit_transform(all_states_color)
#         explained_variance_color = pca_color.explained_variance_ratio_
#         print("Color task: Explained variance ratio of PC1, PC2, PC3:", explained_variance_color)
#         pcs_color_reshaped = pcs_color.reshape(len(indices), T, 3)
#         color_trials = color_trials.copy()
#         color_trials['pcs'] = list(pcs_color_reshaped)
        
#         # Group by color_coherences and compute the average trajectory (averaged over trials)
#         unique_color = sorted(color_trials['color_coherences'].unique())
#         print("Unique color coherences:", unique_color)
        
#         # ----- 2D Projection (PC1 vs PC2) -----
#         plt.figure(figsize=(8, 6))
#         for coherence in unique_color:
#             group_trials = color_trials[color_trials['color_coherences'] == coherence]
#             trajs = np.stack(group_trials['pcs'].values, axis=0)
#             avg_traj = trajs.mean(axis=0)
#             plt.plot(avg_traj[:, 0], avg_traj[:, 1], marker='o', label=f"Color coherence: {coherence}")
#         plt.xlabel("PC1")
#         plt.ylabel("PC2")
#         plt.title("PCA Trajectories for Color Task (context_flags==0)\n(Merged over motion_coherences) - 2D")
#         plt.legend()
#         plt.show()
        
#         # ----- 3D Projection -----
#         from mpl_toolkits.mplot3d import Axes3D
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection='3d')
#         for coherence in unique_color:
#             group_trials = color_trials[color_trials['color_coherences'] == coherence]
#             trajs = np.stack(group_trials['pcs'].values, axis=0)
#             avg_traj = trajs.mean(axis=0)
#             ax.plot(avg_traj[:, 0], avg_traj[:, 1], avg_traj[:, 2], marker='o', label=f"Color coherence: {coherence}")
#         ax.set_xlabel("PC1")
#         ax.set_ylabel("PC2")
#         ax.set_zlabel("PC3")
#         ax.set_title("PCA Trajectories for Color Task (context_flags==0)\n(Merged over motion_coherences) - 3D")
#         ax.legend()
#         plt.show()
#     else:
#         print("No trials found for Color Task (context_flags==0).")

def perform_pca_and_plot(state_trajectories, trial_info):
    """
    对隐藏状态轨迹同时进行 PCA 分析和 regression subspace 分析，并分别绘制 2D 与 3D 投影图。
    
    输入：
      state_trajectories: numpy array, shape = [num_trials, T, hidden_size]
      trial_info: list of dict，每个字典包含 'motion_coherences', 'color_coherences', 'context_flags'
    
    本示例中：
      - PCA 部分：先对各任务（motion / color）分别做 PCA，绘制轨迹（与原函数类似）。
      - Regression subspace 部分：先对全体 trial 做全局 PCA 得到去噪矩阵 D，
        然后对每个时间点、每个任务变量（本例中选取 motion_coherences, color_coherences, context_flags，共3个变量）进行线性回归，
        将回归系数投影到 D 所定义的子空间中，选取“贡献最大”的时刻获得单一回归向量，再对这 3 个向量做 QR 分解得到正交化的回归子空间。
        最后将每个 trial 的状态轨迹投影到该 regression subspace 中，并按照任务参数分组绘图。
    """
    num_trials, T, hidden_size = state_trajectories.shape

    # 构造 trial 信息 DataFrame，并添加 trial 编号
    trial_df = pd.DataFrame(trial_info)
    trial_df['trial'] = np.arange(num_trials)
    # 如果 motion_coherences, color_coherences 是数组，则提取单个数值
    trial_df['motion_coherences'] = trial_df['motion_coherences'].apply(lambda x: x.item() if isinstance(x, np.ndarray) and x.size==1 else x)
    trial_df['color_coherences'] = trial_df['color_coherences'].apply(lambda x: x.item() if isinstance(x, np.ndarray) and x.size==1 else x)
    
    # ============================
    # 第一部分：基于 PCA 的轨迹绘图（分别对 motion 与 color 任务）
    # ============================
    
    # ---- Motion Task（例如 context_flags==0 表示 motion 任务）----
    motion_trials = trial_df[trial_df['context_flags'] == 0]
    if not motion_trials.empty:
        indices = motion_trials['trial'].values
        states_motion = state_trajectories[indices]  # [n_trials_motion, T, hidden_size]
        # 将所有 trial 的状态展开用于 PCA
        all_states_motion = states_motion.reshape(-1, hidden_size)
        pca_motion = PCA(n_components=3)
        pcs_motion = pca_motion.fit_transform(all_states_motion)
        explained_variance_motion = pca_motion.explained_variance_ratio_
        print("Motion task (PCA): Explained variance ratio of PC1, PC2, PC3:", explained_variance_motion)
        # 重塑为 [n_trials_motion, T, 3]
        pcs_motion_reshaped = pcs_motion.reshape(len(indices), T, 3)
        motion_trials = motion_trials.copy()
        motion_trials['pcs'] = list(pcs_motion_reshaped)
        
        # 按 motion_coherences 分组并计算平均轨迹
        unique_motion = sorted(motion_trials['motion_coherences'].unique())
        print("Unique motion coherences:", unique_motion)
        
        # ----- 2D 投影（PC1 vs PC2）-----
        plt.figure(figsize=(8, 6))
        for coherence in unique_motion:
            group_trials = motion_trials[motion_trials['motion_coherences'] == coherence]
            trajs = np.stack(group_trials['pcs'].values, axis=0)  # shape: [n_trials, T, 3]
            avg_traj = trajs.mean(axis=0)  # [T, 3]
            plt.plot(avg_traj[:, 0], avg_traj[:, 1], marker='o', label=f"Motion coherence: {coherence}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Trajectories for Motion Task\n(Merged over color coherences) - 2D")
        plt.legend()
        plt.show()
        
        # ----- 3D 投影 -----
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for coherence in unique_motion:
            group_trials = motion_trials[motion_trials['motion_coherences'] == coherence]
            trajs = np.stack(group_trials['pcs'].values, axis=0)
            avg_traj = trajs.mean(axis=0)
            ax.plot(avg_traj[:, 0], avg_traj[:, 1], avg_traj[:, 2], marker='o', label=f"Motion coherence: {coherence}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("PCA Trajectories for Motion Task\n(Merged over color coherences) - 3D")
        ax.legend()
        plt.show()
    else:
        print("No trials found for Motion Task (context_flags==0).")
    
    # ---- Color Task（例如 context_flags==1 表示 color 任务）----
    color_trials = trial_df[trial_df['context_flags'] == 1]
    if not color_trials.empty:
        indices = color_trials['trial'].values
        states_color = state_trajectories[indices]  # [n_trials_color, T, hidden_size]
        all_states_color = states_color.reshape(-1, hidden_size)
        pca_color = PCA(n_components=3)
        pcs_color = pca_color.fit_transform(all_states_color)
        explained_variance_color = pca_color.explained_variance_ratio_
        print("Color task (PCA): Explained variance ratio of PC1, PC2, PC3:", explained_variance_color)
        pcs_color_reshaped = pcs_color.reshape(len(indices), T, 3)
        color_trials = color_trials.copy()
        color_trials['pcs'] = list(pcs_color_reshaped)
        
        unique_color = sorted(color_trials['color_coherences'].unique())
        print("Unique color coherences:", unique_color)
        
        # ----- 2D 投影（PC1 vs PC2）-----
        plt.figure(figsize=(8, 6))
        for coherence in unique_color:
            group_trials = color_trials[color_trials['color_coherences'] == coherence]
            trajs = np.stack(group_trials['pcs'].values, axis=0)
            avg_traj = trajs.mean(axis=0)
            plt.plot(avg_traj[:, 0], avg_traj[:, 1], marker='o', label=f"Color coherence: {coherence}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Trajectories for Color Task\n(Merged over motion coherences) - 2D")
        plt.legend()
        plt.show()
        
        # ----- 3D 投影 -----
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for coherence in unique_color:
            group_trials = color_trials[color_trials['color_coherences'] == coherence]
            trajs = np.stack(group_trials['pcs'].values, axis=0)
            avg_traj = trajs.mean(axis=0)
            ax.plot(avg_traj[:, 0], avg_traj[:, 1], avg_traj[:, 2], marker='o', label=f"Color coherence: {coherence}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("PCA Trajectories for Color Task\n(Merged over motion coherences) - 3D")
        ax.legend()
        plt.show()
    else:
        print("No trials found for Color Task (context_flags==1).")
    
    # ============================
    # 第二部分：Regression Subspace 分析及轨迹绘图
    # ============================
    # 本例中我们构造一个设计矩阵 Z，包含 4 个变量：
    #   Z[:,0] : motion_coherences
    #   Z[:,1] : color_coherences
    #   Z[:,2] : context_flags
    #   Z[:,3] : target (1 or -1)

    num_vars = 4
    Z = np.stack([trial_df['motion_coherences'].values,
                  trial_df['color_coherences'].values,
                  trial_df['context_flags'].values, 
                  trial_df['target'].values], axis=1)  # shape: [num_trials, 4]
    
    # 为构建全局去噪子空间，先对所有 trial 的状态进行 PCA（参考文章：对 N_units x (N_conditions * T) 进行 PCA）
    X_all = state_trajectories.reshape(-1, hidden_size) # [num_trials * T, hidden_size]
    X_all = (X_all - X_all.mean(axis=0)) / X_all.std(axis=0)
    pca_global = PCA(n_components=12)
    pca_global.fit(X_all)
    V_global = pca_global.components_.T # [hidden_size, 12]
    D = V_global @ V_global.T  # 去噪矩阵 D, shape: [hidden_size, hidden_size]
    
    # 对于每个时间点 t 和每个变量 v，进行线性回归，预测 hidden state（每个 trial 的状态）
    # betas: shape [num_vars, T, hidden_size]
    betas = np.zeros((num_vars, T, hidden_size))
    for t in range(T):
        # X_target: [num_trials, hidden_size]，代表所有 trial 在 t 时刻的状态
        # state_trajectories[:, t, :] 的 shape 为 [num_trials, hidden_size], originally [num_trials, T, hidden_size]
        X_target = state_trajectories[:, t, :] 
        for v in range(num_vars):
            # 对于单变量回归：用 Z[:, v] 预测 X_target
            model = LinearRegression()
            # 注意：Z[:, v] 的形状为 (num_trials,)，需要变成二维数组
            model.fit(Z[:, v].reshape(-1, 1), X_target)
            # model.coef_ 的 shape 为 (hidden_size,) 或 (hidden_size, 1)，这里取 flatten
            betas[v, t, :] = model.coef_.flatten()
    
    # 将每个 beta 向量投影到 PCA 去噪子空间中
    # betas_denoised: shape [num_vars, T, hidden_size]
    betas_denoised = np.einsum('ij,vtj->vti', D, betas)

    for v in range(num_vars):
        plt.plot(np.linalg.norm(betas_denoised[v], axis=1), label=f'Var {v}')
    plt.xlabel("Time")
    plt.ylabel("||beta||")
    plt.title("Norm of beta over time")
    plt.legend()
    plt.show()
    
    # 对每个变量，选择“贡献”最大的时间点（例如向量范数最大的时刻）
    regression_axes = []
    for v in range(num_vars):
        norms = np.linalg.norm(betas_denoised[v], axis=1)  # [T,]
        best_t = np.argmax(norms)
        regression_axes.append(betas_denoised[v, best_t, :])
    # 将得到的 regression 向量构成矩阵 B：shape [hidden_size, num_vars]
    B = np.stack(regression_axes, axis=1)
    # 对 B 进行 QR 分解，得到正交的回归 subspace 基底
    Q, R = qr(B, mode='economic')  # Q: [hidden_size, num_vars]
    print("QR decomposition: Q shape =", Q.shape, ", R shape =", R.shape)
    
    # 对每个 trial 的状态轨迹投影到 regression subspace（即 Q 所张成的子空间）
    # 得到投影轨迹：projected_states, shape: [num_trials, T, num_vars]
    projected_states = state_trajectories @ Q
    
    # 分别对 Motion Task 与 Color Task 绘制 regression subspace 下的轨迹
    # ---- Motion Task (context_flags==0) ----
    if not motion_trials.empty:
        indices = motion_trials['trial'].values
        proj_motion = projected_states[indices]  # shape: [n_trials_motion, T, num_vars]
        proj_motion_reshaped = proj_motion  # 已经是 3 维（num_vars==3）
        motion_trials = motion_trials.copy()
        motion_trials['proj'] = list(proj_motion_reshaped)
        
        # 按 motion_coherences 分组，计算平均轨迹
        plt.figure(figsize=(8, 6))
        for coherence in unique_motion:
            group_trials = motion_trials[motion_trials['motion_coherences'] == coherence]
            trajs = np.stack(group_trials['proj'].values, axis=0)  # [n_trials, T, 3]
            avg_traj = trajs.mean(axis=0)
            plt.plot(avg_traj[:, 0], avg_traj[:, 1], marker='o', label=f"Motion coherence: {coherence}")
        plt.xlabel("Regression Axis 1")
        plt.ylabel("Regression Axis 2")
        plt.title("Regression Subspace Trajectories for Motion Task\n(Projection onto Q: first 2 axes) - 2D")
        plt.legend()
        plt.show()
        
        # 3D 绘图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for coherence in unique_motion:
            group_trials = motion_trials[motion_trials['motion_coherences'] == coherence]
            trajs = np.stack(group_trials['proj'].values, axis=0)
            avg_traj = trajs.mean(axis=0)
            ax.plot(avg_traj[:, 0], avg_traj[:, 1], avg_traj[:, 2], marker='o', label=f"Motion coherence: {coherence}")
        ax.set_xlabel("Axis 1")
        ax.set_ylabel("Axis 2")
        ax.set_zlabel("Axis 3")
        ax.set_title("Regression Subspace Trajectories for Motion Task - 3D")
        ax.legend()
        plt.show()
    else:
        print("No trials found for Motion Task (context_flags==0) in regression subspace analysis.")
    
    # ---- Color Task (context_flags==1) ----
    if not color_trials.empty:
        indices = color_trials['trial'].values
        proj_color = projected_states[indices]  # shape: [n_trials_color, T, 3]
        proj_color_reshaped = proj_color
        color_trials = color_trials.copy()
        color_trials['proj'] = list(proj_color_reshaped)
        
        plt.figure(figsize=(8, 6))
        for coherence in unique_color:
            group_trials = color_trials[color_trials['color_coherences'] == coherence]
            trajs = np.stack(group_trials['proj'].values, axis=0)
            avg_traj = trajs.mean(axis=0)
            plt.plot(avg_traj[:, 0], avg_traj[:, 1], marker='o', label=f"Color coherence: {coherence}")
        plt.xlabel("Regression Axis 1")
        plt.ylabel("Regression Axis 2")
        plt.title("Regression Subspace Trajectories for Color Task\n(Projection onto Q: first 2 axes) - 2D")
        plt.legend()
        plt.show()
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for coherence in unique_color:
            group_trials = color_trials[color_trials['color_coherences'] == coherence]
            trajs = np.stack(group_trials['proj'].values, axis=0)
            avg_traj = trajs.mean(axis=0)
            ax.plot(avg_traj[:, 0], avg_traj[:, 1], avg_traj[:, 2], marker='o', label=f"Color coherence: {coherence}")
        ax.set_xlabel("Axis 1")
        ax.set_ylabel("Axis 2")
        ax.set_zlabel("Axis 3")
        ax.set_title("Regression Subspace Trajectories for Color Task - 3D")
        ax.legend()
        plt.show()
    else:
        print("No trials found for Color Task (context_flags==1) in regression subspace analysis.")

# ===============================
# Main execution
# ===============================
if __name__ == "__main__":
    exp_path = os.path.join(os.getcwd(), 'results', 'exp_1')
    # Load the model
    model = RNNModel()  # Assume RNNModel is defined elsewhere
    model.load_state_dict(torch.load(os.path.join(exp_path, 'model.pth')))
    model.eval()
    
    # Test the model
    test_model(model, num_trials=100)  # Assume test_model is defined elsewhere
    
    # Record state trajectories from multiple trials (e.g., 50 trials, each with T timesteps)
    state_trajectories, trial_info = record_trajectories(model, num_trials=1000)
    
    # Perform PCA on state trajectories and plot both 2D and 3D projections to help understand network dynamics
    perform_pca_and_plot(state_trajectories, trial_info)