import torch

class Smoother:
    def __init__(self, initial_action, process_noise=0.01, measurement_noise=0.1, device='cpu'):
        self.device = device
        
        # 状态：动作值（假设为4维）
        assert len(initial_action.shape) == 1, "Initial action must be a 1D tensor."
        self.state_dim = initial_action.shape[0]  # 4维动作
        self.x = initial_action.view(-1, 1).float().to(device)  # [4,1] 初始状态
        
        # 状态转移矩阵（简单模型：动作缓慢变化）
        self.F = torch.eye(self.state_dim, device=device)  # [4,4]
        
        # 观测矩阵（直接观测动作）
        self.H = torch.eye(self.state_dim, device=device)  # [4,4]
        
        # 过程噪声（越小越平滑，但可能滞后）
        self.Q = torch.eye(self.state_dim, device=device) * process_noise  # [4,4]
        
        # 观测噪声（越大越信任历史状态）
        self.R = torch.eye(self.state_dim, device=device) * measurement_noise  # [4,4]
        
        # 状态协方差
        self.P = torch.eye(self.state_dim, device=device)  # [4,4]

    def smooth(self, raw_action):
        # 确保输入是torch tensor并正确形状 [4] -> [4,1]
        z = raw_action.view(-1, 1).float().to(self.device)
        
        # 预测步骤
        self.x = self.F @ self.x  # [4,4] @ [4,1] -> [4,1]
        self.P = self.F @ self.P @ self.F.T + self.Q  # [4,4]
        
        # 更新步骤
        y = z - self.H @ self.x  # 残差 [4,1]
        S = self.H @ self.P @ self.H.T + self.R  # [4,4]
        
        # 使用伪逆提高数值稳定性
        K = self.P @ self.H.T @ torch.linalg.pinv(S)  # 卡尔曼增益 [4,4]
        
        self.x = self.x + K @ y  # [4,1]
        self.P = (torch.eye(self.state_dim, device=self.device) - K @ self.H) @ self.P  # [4,4]
        
        return self.x.squeeze()  # 返回平滑后的动作 [4]

# 使用示例
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 模拟神经网络输出的噪声动作（10个时间步，每个动作4维）
    true_value = torch.ones(4)
    noisy_actions = true_value + torch.randn(10, 4) * 0.5  # 添加高斯噪声
    
    # 初始化平滑器（传入第一个动作）
    smoother = Smoother(
        initial_action=noisy_actions[0],
        process_noise=0.01,
        measurement_noise=0.1,
        device=device
    )
    
    # 存储结果
    smoothed_actions = torch.zeros_like(noisy_actions)
    
    # 平滑处理
    for i, action in enumerate(noisy_actions):
        smoothed_actions[i] = smoother.smooth(action)
    
    # 可视化
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
    for i in range(4):
        axes[i].plot(noisy_actions[:, i].cpu().numpy(), 'r-', label='Noisy Actions')
        axes[i].plot(smoothed_actions[:, i].cpu().numpy(), 'g-', label='Smoothed Actions')
        axes[i].axhline(true_value[i], color='b', linestyle='--', label='True Value')
        axes[i].set_title(f'Dimension {i+1}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()