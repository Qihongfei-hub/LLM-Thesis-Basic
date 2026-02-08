
1 Attention Is All You Need (Vaswani et al., 2017) 《注意力就是你需要的一切》 Transformer 原论文.系统介绍了自注意力,多头注意力以及编码器-解码器架构（尽管现代大多数大语言模型采用 decoder-only 结构）. 


2 The Illustrated Transformer (Jay Alammar, 2018) 《图解 Transformer》 极佳的直觉构建材料，帮助在阅读实现代码前深入理解注意力机制和张量流动。

3 BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018) 《BERT：深度双向 Transformer 的预训练》 编码器侧基础，掩码语言建模（MLM）和表示学习，至今仍深刻影响现代架构设计。

4 Language Models are Few-Shot Learners (Brown et al., 2020) 《语言模型是少样本学习者》（GPT-3 论文） 确立了上下文学习（in-context learning）的真实能力，彻底改变了人们对提示（prompting）的理解。

5 Scaling Laws for Neural Language Models (Kaplan et al., 2020) 《神经语言模型的缩放定律》 首次提出参数、数据、计算的干净经验缩放框架。建议与 Chinchilla 论文一起阅读，以理解为何多数模型训练不足。

6 Training Compute-Optimal Large Language Models (Hoffmann et al., 2022) 《训练计算最优的大型语言模型》（Chinchilla 论文） 证明在固定计算预算下，token 数量比参数数量更重要。

7 LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023) 《LLaMA：开放高效的基础语言模型》 开启开源权重时代的标志性论文。将 RMSNorm、SwiGLU、RoPE 等设为现代架构默认标准。

8 RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021) 《RoFormer：增强型 Transformer 采用旋转位置嵌入》 提出旋转位置嵌入（RoPE），成为长上下文大模型的现代默认位置编码方式。

9 FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al., 2022) 《FlashAttention：快速且内存高效的精确注意力》 通过优化 GPU 内存访问，实现内存高效注意力，支持长上下文窗口和高吞吐推理。

10 Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020) 《检索增强生成》（RAG 论文） 将参数模型与外部知识源结合，是接地（grounded）和企业级系统的基石。

11 Training Language Models to Follow Instructions with Human Feedback (Ouyang et al., 2022) 《使用人类反馈训练语言模型遵循指令》（InstructGPT 论文） 现代后训练与对齐的蓝图，当前所有指令微调模SFT型基本遵循此范式。

12 Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023) 《直接偏好优化》（DPO 论文） 比基于 PPO 的 RLHF 更简单、更稳定的偏好对齐方法，直接通过损失函数实现。

13 Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022) 《思维链提示引发大语言模型中的推理》（CoT 论文） 证明仅通过提示即可激发推理能力，为后续专注推理的训练方法奠定基础。

14 ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022 / ICLR 2023) 《ReAct：语言模型中推理与行动的协同》 Agentic 系统的基石，将推理轨迹与工具使用、环境交互相结合。

15 DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (Guo et al., 2025) 《DeepSeek-R1：通过强化学习激励大语言模型的推理能力》 R1 论文。证明大规模强化学习无需监督数据即可诱导自验证和结构化推理行为。

16 Qwen3 Technical Report (Yang et al., 2025) 《Qwen3 技术报告》 现代架构的轻量级概览。引入统一 MoE 的思考模式与非思考模式，动态权衡成本与推理深度。

17 Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (Shazeer et al., 2017) 《超大规模神经网络：稀疏门控混合专家层》 现代 MoE 的点火之作，实现大规模条件计算。

18 Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (Fedus et al., 2021) 《Switch Transformers：大规模稀疏语言模型的缩放》 采用单专家激活简化 MoE 路由，是稳定训练万亿参数模型的关键。

19 Mixtral of Experts (Mistral AI, 2024) 《Mixtral of Experts》 开源权重 MoE 模型，证明稀疏模型可在小模型推理成本下匹配稠密模型质量。

20 Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints (Komatsuzaki et al., 2022 / ICLR 2023) 《稀疏上行循环：从稠密检查点训练混合专家模型》 将稠密检查点转换为 MoE 模型的实用技术，对计算复用和迭代缩放至关重要。

21 The Platonic Representation Hypothesis (Huh et al., 2024) 《柏拉图表示假设》 提供证据表明，随着模型规模扩大，不同模态的内部表示趋于收敛到共享形式。

22 Textbooks Are All You Need (Gunasekar et al., 2023) 《教科书就是你需要的一切》 证明高质量合成数据可使小模型大幅超越远大于自身的模型。  // 蒸馏??

23 Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet (Templeton et al., 2024) 《单义性缩放：从 Claude 3 Sonnet 中提取可解释特征》 机制可解释性领域的最大突破，将神经网络分解为数百万个可解释特征。

24 PaLM: Scaling Language Modeling with Pathways (Chowdhery et al., 2022) 《PaLM：使用 Pathways 缩放大语言建模》 大规模训练编排的教科书级案例，跨越数千加速器。

25 GLaM: Efficient Scaling of Language Models with Mixture-of-Experts (Du et al., 2022) 《GLaM：专家混合高效大型语言模型》 用极大量总参数但少量活跃参数验证 MoE 缩放的经济性。




附加阅读材料
 • The Smol Training Playbook (Hugging Face, 2025) 《Smol 训练手册》 高效训练语言模型的端到端实用指南.

• T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (Raffel et al., 2019) 《T5：用统一的文本到文本 Transformer 探索迁移学习的极限》

• Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023) 《Toolformer：语言模型通过 API 调用自学工具》

• GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding (Lepikhin et al., 2020) 《GShard：大规模稀疏模型的并行训练》

• Adaptive Mixture of Local Experts (Jacobs et al., 1991) 《自适应局部专家混合》

• Hierarchical Mixture of Experts (Jordan and Jacobs, 1994) 《层次专家混合》

