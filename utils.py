import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_context_initial_state(model, context_flag, steps=200):
    """
    根据 context_flag (0: motion, 1: color) 生成初始状态，
    利用固定的 context 输入（仅 context 信号为1，其余为0）使网络收敛到稳定状态。
    """
    device = next(model.parameters()).device
    x = torch.zeros(1, model.hidden_size, device=device)
    # construct context input
    if context_flag == 0:
        u_const = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device)
    else:
        u_const = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            r = torch.tanh(x)
            input_term = (u_const[:, 0:1] @ model.b_m.t() +
                          u_const[:, 1:2] @ model.b_c.t() +
                          u_const[:, 2:3] @ model.b_cm.t() +
                          u_const[:, 3:4] @ model.b_cc.t())
            dx = (model.dt / model.tau) * (-x + model.J(torch.tanh(x)) + input_term + model.c_x)
            x = x + dx
    return x  # shape [1, hidden_size]

def refine_slow_point(model, context_flag, steps=200, refine_steps=200, lr=1e-2):
    """
    在给定 context_flag 下进一步寻找使得网络输出接近零的 slow point，
    该 slow point 应同时接近固定点且使输出 z 接近0。
    """
    device = next(model.parameters()).device
    # 先用 get_context_initial_state 得到一个初始解
    x_init = get_context_initial_state(model, context_flag, steps=steps)
    x = x_init.clone().requires_grad_(True)
    
    # construct constant input of context
    if context_flag == 0:
        u_const = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device)
    else:
        u_const = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
        
    optimizer = torch.optim.Adam([x], lr=lr)
    
    model.eval()
    for _ in range(refine_steps):
        optimizer.zero_grad()
        r = torch.tanh(x)
        input_term = (u_const[:, 0:1] @ model.b_m.t() +
                      u_const[:, 1:2] @ model.b_c.t() +
                      u_const[:, 2:3] @ model.b_cm.t() +
                      u_const[:, 3:4] @ model.b_cc.t())
        # 定义动力学残差：f(x) = -x + J*tanh(x) + input_term + c_x
        f_x = -x + model.J(r) + input_term + model.c_x
        # 输出 z = W_out * tanh(x)
        z = model.W_out(r)
        # 总 loss：动力学残差 + 输出幅值（使输出接近零），权重 0.1 可调
        loss = f_x.pow(2).mean() + 0.1 * z.pow(2).mean()
        loss.backward()
        optimizer.step()
    return x.detach()  # return the refined slow point