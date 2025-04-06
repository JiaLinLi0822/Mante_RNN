import numpy as np
from scipy.optimize import minimize
from numpy.linalg import eig
import torch
import os
from env import *
from net import *

def fixed_point_analysis_rnn(model, u, num_starts=100, tol=1e-6):
    """
    对给定 RNNModel 模型在固定输入 u 下进行固定点分析。
    
    参数：
      model: 已训练好的 RNNModel 模型（处于 eval 模式）
      u: 固定输入向量，形状为 (4,)，对应 [u_m, u_c, u_cm, u_cc]
      num_starts: 随机起点数
      tol: 判断固定点的误差容忍度

    返回：
      fixed_points: list，每个元素为一个固定点向量（numpy 数组，形状 (hidden_size,)）
      dynamics: list，每个元素是一个字典，包含该固定点的雅可比矩阵及特征值特征向量
    """
    # 提取模型参数并转为 numpy 数组
    hidden_size = model.hidden_size
    
    # Recurrent weight matrix, shape: [hidden_size, hidden_size]
    J = model.J.weight.detach().cpu().numpy()  # 无偏置
    
    # 输入权重参数，均为 shape: [hidden_size, 1]
    b_m = model.b_m.detach().cpu().numpy().flatten()   # motion evidence
    b_c = model.b_c.detach().cpu().numpy().flatten()   # color evidence
    b_cm = model.b_cm.detach().cpu().numpy().flatten() # motion context
    b_cc = model.b_cc.detach().cpu().numpy().flatten() # color context
    
    # 状态偏置， shape: (hidden_size,)
    c_x = model.c_x.detach().cpu().numpy()
    
    # 根据固定输入 u 计算输入项 (忽略 batch 维度, 只针对单个样本)
    # u: array of shape (4,), u[0]=u_m, u[1]=u_c, u[2]=u_cm, u[3]=u_cc
    input_term = u[0] * b_m + u[1] * b_c + u[2] * b_cm + u[3] * b_cc
    # 总输入项（含偏置）
    b_in = input_term + c_x

    # 固定点方程: f(x) = -x + J * tanh(x) + b_in, fixed point满足 f(x)=0
    def f(x):
        return -x + J @ np.tanh(x) + b_in

    def loss(x):
        return 0.5 * np.sum(f(x)**2)
    
    fixed_points = []
    dynamics = []
    
    for i in range(num_starts):
        # 随机初始化一个 x0
        x0 = np.random.randn(hidden_size)
        res = minimize(loss, x0, method='BFGS')
        if res.success and loss(res.x) < tol:
            x_star = res.x
            # 检查是否已找到相近的固定点
            if not any(np.allclose(x_star, fp, atol=1e-3) for fp in fixed_points):
                fixed_points.append(x_star)
                # 计算固定点处的雅可比矩阵
                # 对于 f(x) = -x + J*tanh(x) + b_in, 雅可比 df/dx = -I + J*diag(1 - tanh(x)^2)
                phi_prime = 1 - np.tanh(x_star)**2  # elementwise derivative of tanh
                M = -np.eye(hidden_size) + J @ np.diag(phi_prime)
                eigvals, eigvecs = eig(M)
                dynamics.append({
                    "x_star": x_star,
                    "jacobian": M,
                    "eigvals": eigvals,
                    "eigvecs": eigvecs
                })
                print(f"Found fixed point #{len(fixed_points)} at iteration {i+1}, loss = {loss(x_star):.2e}")
    
    return fixed_points, dynamics

# 示例：使用训练好的模型对固定输入进行固定点分析
if __name__ == "__main__":

    exp_path = os.path.join(os.getcwd(), 'results', 'exp_1')
    # Load the model
    model = RNNModel()  # Assume RNNModel is defined elsewhere
    model.load_state_dict(torch.load(os.path.join(exp_path, 'model.pth')))
    model.eval()

    # 固定输入 u, 例如设为 [0.5, 0.0, 1.0, 0.0] （你可根据需要调整）
    u_fixed = np.array([0.5, 0.0, 1.0, 0.0])
    
    fixed_pts, dyn = fixed_point_analysis_rnn(model, u_fixed, num_starts=50, tol=1e-6)
    print(f"Found {len(fixed_pts)} fixed points.")
    
    # 打印每个固定点的最大实部特征值，判断稳定性
    for i, d in enumerate(dyn):
        max_real = np.max(np.real(d["eigvals"]))
        print(f"Fixed point #{i+1}: Max real part of eigenvalues = {max_real:.3f}")