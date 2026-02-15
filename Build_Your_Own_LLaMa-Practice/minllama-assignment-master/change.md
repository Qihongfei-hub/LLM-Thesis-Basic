# 代码改动记录

## 1. llama.py

### 改动内容

#### 1.1 RMSNorm._norm 方法
- **实现功能**：根均方归一化
- **实现细节**：
  - 使用论文公式计算根均方归一化
  - 添加 epsilon 值以确保数值稳定性
  - 代码：`norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps); return x / norm`

#### 1.2 Attention.compute_query_key_value_scores 方法
- **实现功能**：缩放点积注意力
- **实现细节**：
  - 同时计算多个头的注意力
  - 按照论文要求对注意力分数进行缩放
  - 应用了注意力 dropout 以防止过拟合
  - 计算注意力分数、缩放、应用 softmax、应用 dropout、计算加权和

#### 1.3 LlamaLayer.forward 方法
- **实现功能**：transformer 基本构建块的前向传播
- **实现细节**：
  - 包括层归一化、自注意力、残差连接和前馈网络
  - 按照现代 transformer 架构实现了完整的前向传播流程

#### 1.4 Llama.generate 方法
- **实现功能**：生成过程，使用温度采样
- **实现细节**：
  - 处理了温度为 0 的情况，选择最可能的 token
  - 实现了完整的温度采样流程：缩放 logits、应用 softmax、从概率分布中采样

## 2. classifier.py

### 改动内容

#### 2.1 LlamaEmbeddingClassifier.forward 方法
- **实现功能**：分类器前向传播
- **实现细节**：
  - 获取 llama 模型的隐藏状态
  - 找到输入序列最后一个 token 的隐藏状态
  - 应用 dropout 以减少过拟合
  - 通过分类头获取 logits
  - 计算 log-softmax 并返回 log-概率

## 3. optimizer.py

### 改动内容

#### 3.1 AdamW.step 方法
- **实现功能**：AdamW 优化器的参数更新步骤
- **实现细节**：
  - 初始化状态字典，存储步骤计数器、梯度的一阶矩和二阶矩
  - 获取超参数，包括学习率、betas、epsilon、权重衰减和是否进行偏置校正
  - 更新梯度的一阶矩和二阶矩
  - 进行偏置校正，使用论文中的高效版本
  - 更新参数，使用计算出的步长
  - 在主要的基于梯度的更新后添加权重衰减

## 验证结果

- 使用 GetDiagnostics 工具检查代码，返回空列表，说明代码中没有语法错误或类型错误
- 所有代码实现都遵循了注释中的要求，使用了相关论文中描述的算法和公式
