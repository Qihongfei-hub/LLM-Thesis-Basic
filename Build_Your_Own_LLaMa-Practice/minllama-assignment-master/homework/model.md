Transformer模型的层数和每层参数：

### 1. 模型层数
- 在 Llama 类的 __init__ 方法中（第233行）： self.n_layers = config.n_layers
- 第238-239行通过循环创建对应数量的层：
  ```
  for layer_id in range(config.n_layers):
      self.layers.append(LlamaLayer(layer_id, 
      config))
  ```
- 实际层数由预训练模型配置决定，从 stories42M.pt 文件加载
### 2. 每层结构（LlamaLayer）
每个 LlamaLayer 包含以下组件：

- 注意力模块 （Attention）：
  
  - compute_query : 线性层，输入维度 config.dim ，输出维度 config.n_heads * head_dim
  - compute_key : 线性层，输入维度 config.dim ，输出维度 n_kv_heads * head_dim
  - compute_value : 线性层，输入维度 config.dim ，输出维度 n_kv_heads * head_dim
  - compute_output : 线性层，输入维度 config.n_heads * head_dim ，输出维度 config.dim
  - 两个 dropout 层
- 前馈网络 （FeedForward）：
  
  - w1 : 线性层，输入维度 dim ，输出维度 hidden_dim
  - w2 : 线性层，输入维度 hidden_dim ，输出维度 dim
  - w3 : 线性层，输入维度 dim ，输出维度 hidden_dim
  - 一个 dropout 层
- 归一化层 ：
  
  - attention_norm : RMSNorm 层，包含 config.dim 个可学习参数
  - ffn_norm : RMSNorm 层，包含 config.dim 个可学习参数
### 3. 整体模型结构
- 嵌入层 ： tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
- Transformer层 ： n_layers 个 LlamaLayer
- 输出层 ： output = nn.Linear(config.dim, config.vocab_size, bias=False)
  - 注意：输出层与嵌入层共享参数（第244行）