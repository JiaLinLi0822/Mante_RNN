import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from env import *
from net import *

def test_model(model, num_trials=1000, T=20, device='cpu'):
    model.eval()
    correct = 0
    env = RDM()
    with torch.no_grad():
        for _ in range(num_trials):
            input_seq, targets = env.generate_trial(batch_size=1)
            output = model(input_seq)
            pred = 1.0 if output.item() >= 0 else -1.0
            if pred == targets.item():
                correct += 1
    accuracy = correct / num_trials
    print(f"Test Accuracy over {num_trials} trials: {accuracy*100:.2f}%")
    return accuracy

# ===============================
# 记录状态轨迹
# ===============================
def record_trajectories(model, num_trials=50, T=20, device='cpu'):
    """
    对多次试次运行记录网络隐藏状态的轨迹。
    返回：形状为 [num_trials, T, hidden_size] 的 numpy 数组
    """
    model.eval()
    all_states = []
    env = RDM()
    with torch.no_grad():
        for i in range(num_trials):
            input_seq, _ = env.generate_trial(batch_size=1)
            # 调用 forward_with_states 得到每个时间步的隐藏状态，shape: [T, 1, hidden_size]
            states = model.forward_with_states(input_seq)
            # 去掉 batch 维度 -> [T, hidden_size]
            states = states.squeeze(1)
            all_states.append(states.cpu().numpy())
    all_states = np.array(all_states)  # shape: [num_trials, T, hidden_size]
    return all_states

# ===============================
# 对状态轨迹进行 PCA 降维并绘图
# ===============================
def perform_pca_and_plot(state_trajectories, num_trials_to_plot=10):
    """
    对记录的隐藏状态轨迹进行 PCA 分析，并绘制 2D 与 3D 投影图。
    参数：
       state_trajectories: [num_trials, T, hidden_size]
    """
    num_trials, T, hidden_size = state_trajectories.shape
    # 将所有状态点平铺成矩阵，形状为 [num_trials*T, hidden_size]
    all_states = state_trajectories.reshape(-1, hidden_size)
    
    # 进行 PCA 降维至 3 个主成分
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(all_states)
    explained_variance = pca.explained_variance_ratio_
    print("Explained variance ratio of PC1, PC2, PC3:", explained_variance)
    
    # 将降维后的数据还原为 [num_trials, T, 3]
    pcs_reshaped = pcs.reshape(num_trials, T, 3)
    
    # --- 绘制 2D 投影（PC1 vs PC2） ---
    plt.figure(figsize=(8,6))
    for i in range(min(num_trials_to_plot, num_trials)):
        plt.plot(pcs_reshaped[i, :, 0], pcs_reshaped[i, :, 1], marker='o', label=f"Trial {i+1}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("State trajectories in PCA space (PC1 vs PC2)")
    plt.legend()
    plt.show()
    
    # --- 绘制 3D 投影 ---
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(min(num_trials_to_plot, num_trials)):
        ax.plot(pcs_reshaped[i, :, 0], pcs_reshaped[i, :, 1], pcs_reshaped[i, :, 2],
                marker='o', label=f"Trial {i+1}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("State trajectories in PCA space (3D)")
    plt.legend()
    plt.show()

def train_model(num_epochs=1000, batch_size=64, T=20, learning_rate=1e-3, device='cpu'):
    """
    训练连续时间 RNN 模型，只使用 MSE 作为损失函数，不使用 BPTT。
    
    参数：
      num_epochs   : 训练周期数
      batch_size   : 每次训练的试次数
      T            : 每个试次的时间步数（该参数目前未使用）
      learning_rate: 学习率
      device       : 'cpu' 或 'cuda'
    
    返回：
      model, loss_history
    """
    model = RNNModel(input_size=4, hidden_size=100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    env = RDM()
    
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        input_seq, targets = env.generate_trial(batch_size)
        optimizer.zero_grad()
        
        # 直接通过模型获得输出，并计算 MSE 损失
        outputs = model(input_seq).squeeze()
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")
            
    return model, loss_history

if __name__ == "__main__":
    # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, loss_history = train_model(num_epochs=1000, batch_size=64, T=100, learning_rate=1e-3, device=device)
    
    # 绘制损失曲线
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.show()
    
    # 测试模型
    test_model(model, num_trials=1000, T=100, device=device)
    
    # 记录多个试次的隐藏状态轨迹（例如 50 个试次，每个试次 T 个时间步）
    state_trajectories = record_trajectories(model, num_trials=50, T=100, device=device)
    
    # 利用 PCA 对状态轨迹进行降维，并绘制 2D/3D 投影图，帮助理解网络动态
    perform_pca_and_plot(state_trajectories, num_trials_to_plot=10)