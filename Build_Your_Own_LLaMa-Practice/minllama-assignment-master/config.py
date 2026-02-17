from typing import Union, Tuple, Dict, Any, Optional
import os
import json
from collections import OrderedDict
import torch
from utils import CONFIG_NAME, hf_bucket_url, cached_path, is_remote_url

class PretrainedConfig(object):
  """
  Base class for all configuration classes.
  
  Args:
      model_type (str): Model type identifier.
      is_composition (bool): Whether this configuration corresponds to a composite model.
  """
  
  model_type: str = ""
  is_composition: bool = False

  def __init__(self, **kwargs):
    # Attributes with defaults
    self.return_dict = kwargs.pop("return_dict", True)
    self.output_hidden_states = kwargs.pop("output_hidden_states", False)
    self.output_attentions = kwargs.pop("output_attentions", False)
    self.torchscript = kwargs.pop("torchscript", False)  # Only used by PyTorch models
    self.use_bfloat16 = kwargs.pop("use_bfloat16", False)
    self.pruned_heads = kwargs.pop("pruned_heads", {})
    self.tie_word_embeddings = kwargs.pop(
      "tie_word_embeddings", True
    )  # Whether input and output word embeddings should be tied for all MLM, LM and Seq2Seq models.
    '''
    参数的核心目的是 决定是否共享输入和输出词嵌入权重 。当设置为 True 时，模型会使用相同的权重矩阵进行：
    输入词嵌入 ：将输入 tokens 转换为向量表示
    输出词嵌入 ：从隐藏状态映射回词汇表概率分布
    '''

    # Is decoder is used in encoder-decoder models to differentiate encoder from decoder
    self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
    self.is_decoder = kwargs.pop("is_decoder", False)
    self.add_cross_attention = kwargs.pop("add_cross_attention", False)
    self.tie_encoder_decoder = kwargs.pop("tie_encoder_decoder", False)
    #in this project,decode only is used, refer setting in llama.py

    # Parameters for sequence generation
    self.max_length = kwargs.pop("max_length", 20)
    '''
    参数的核心目的是 控制生成文本的最大长度 。当设置为一个具体的整数时，模型会在生成文本时限制其长度不超过该值。
    这在确保生成文本的可读性和连贯性方面非常重要，避免生成过长的文本。
    '''
    self.min_length = kwargs.pop("min_length", 0)
    self.do_sample = kwargs.pop("do_sample", False)
    '''
    参数的核心目的是 控制是否使用 采样策略 生成文本 。当设置为 True 时，模型会根据其预测的概率分布随机采样 tokens ，
    而不是选择概率最高的 token 。这可以引入更多的多样性和 creativity ，但也可能导致生成文本的质量下降。
    '''
    self.early_stopping = kwargs.pop("early_stopping", False)
    '''
    参数的核心目的是 控制是否在生成文本时 早停 。当设置为 True 时，模型会在生成文本时根据验证集指标（如 BLEU 分数）
    自动停止生成，避免生成过长的文本而导致质量下降。
    '''
    self.num_beams = kwargs.pop("num_beams", 1)
    '''
    参数的核心目的是 控制生成文本时使用的 束搜索 数量 。当设置为大于 1 的整数时，模型会在生成文本时考虑多个可能的序列，
    并选择其中概率最高的 num_beams 个序列作为最终输出。这可以提高生成文本的质量和连贯性，但也会增加计算成本。
    '''
    self.num_beam_groups = kwargs.pop("num_beam_groups", 1)
    '''
    参数的核心目的是 控制生成文本时使用的 束搜索 分组数量 。当设置为大于 1 的整数时，模型会将 num_beams 个序列
    分为 num_beam_groups 组，每组独立考虑，最后合并为最终输出。这可以用于 引入更多的多样性 ，但也会增加计算成本。
    '''
    self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
    self.temperature = kwargs.pop("temperature", 1.0)
    '''
    参数的核心目的是 控制生成文本时使用的 采样温度 。当设置为大于 1 的值时，模型会生成更随机的文本，
    而当设置为小于 1 的值时，模型会生成更确定的文本。这可以用于调整生成文本的多样性和质量。
    '''
    self.top_k = kwargs.pop("top_k", 50)
    self.top_p = kwargs.pop("top_p", 1.0)
    #top_k 和 top_p 是语言模型生成文本时使用的 采样策略参数 ，用于控制生成文本的多样性和质量
    #top_k ：仅考虑概率最高的 k 个 tokens 进行采样
    #top_p ：从概率分布中累积概率直到超过 p ，然后采样其中的 tokens
    self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
    self.length_penalty = kwargs.pop("length_penalty", 1.0)
    '''
    参数的核心目的是 控制生成文本时使用的 长度惩罚 。当设置为大于 1 的值时，模型会在生成文本时考虑其长度，
    并根据长度惩罚因子调整概率分布。这可以用于 控制生成文本的长度 ，避免生成过长的文本。
    '''
    self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
    '''
    参数的核心目的是 控制生成文本时不重复出现的 n-gram 大小 。当设置为大于 0 的整数时，模型会在生成文本时避免重复出现
    相同的 n-gram ，从而提高生成文本的质量和连贯性。
    '''
    self.encoder_no_repeat_ngram_size = kwargs.pop("encoder_no_repeat_ngram_size", 0)
    self.bad_words_ids = kwargs.pop("bad_words_ids", None)
    '''
    参数的核心目的是 控制生成文本时不允许出现的 不良 tokens 列表 。当设置为一个包含不良 tokens ID 的列表时，
    模型会在生成文本时避免出现这些 tokens ，从而提高生成文本的质量和连贯性。
    '''
    self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
    '''
    参数的核心目的是 控制生成文本时返回的 序列数量 。当设置为大于 1 的整数时，模型会在生成文本时返回多个不同的序列，
    每个序列都有不同的概率分布。这可以用于 引入更多的多样性 ，但也会增加计算成本。
    '''
    self.chunk_size_feed_forward = kwargs.pop("chunk_size_feed_forward", 0)
    '''
    参数的核心目的是 控制生成文本时使用的 前馈神经网络 分块大小 。当设置为大于 0 的整数时，模型会将输入序列
    分为多个分块，每个分块独立处理，最后合并为最终输出。这可以用于 处理长序列 ，避免内存溢出问题。
    '''
    self.output_scores = kwargs.pop("output_scores", False) 
    self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)
    self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
    '''
    参数的核心目的是 控制生成文本时强制使用的 开始 tokens ID 。当设置为一个整数时，模型会在生成文本时强制使用该 tokens ID 作为开始 tokens ，
    从而确保生成的文本符合预期的格式。
    '''
    self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
    '''
    参数的核心目的是 控制生成文本时强制使用的 结束 tokens ID 。当设置为一个整数时，模型会在生成文本时强制使用该 tokens ID 作为结束 tokens ，
    从而确保生成的文本符合预期的格式。
    '''

    # Fine-tuning task arguments
    self.architectures = kwargs.pop("architectures", None)
    self.finetuning_task = kwargs.pop("finetuning_task", None)
    '''
    参数的核心目的是 控制微调任务的 类型 。当设置为一个字符串时，模型会根据该字符串选择合适的微调任务，
    并根据任务类型调整模型的输出层。这可以用于 支持不同的 NLP 任务 ，如 文本分类 、 序列标注 等。
    '''
    self.id2label = kwargs.pop("id2label", None)
    '''
    参数的核心目的是 控制微调任务的 标签到 ID 的映射 。当设置为一个字典时，模型会根据该字典将标签映射到对应的 ID ，
    并根据 ID 调整模型的输出层。这可以用于 支持不同的 NLP 任务 ，如 文本分类 、 序列标注 等。
    '''
    self.label2id = kwargs.pop("label2id", None)
    if self.id2label is not None:
      kwargs.pop("num_labels", None)
      self.id2label = dict((int(key), value) for key, value in self.id2label.items())
      # Keys are always strings in JSON so convert ids to int here.
      #将 id2label 字典中的键（key）从字符串类型转换为整数类型 。
    else:
      self.num_labels = kwargs.pop("num_labels", 2)

    # Tokenizer arguments
    self.tokenizer_class = kwargs.pop("tokenizer_class", None)
    self.prefix = kwargs.pop("prefix", None)
    self.bos_token_id = kwargs.pop("bos_token_id", None)
    self.pad_token_id = kwargs.pop("pad_token_id", None)
    self.eos_token_id = kwargs.pop("eos_token_id", None)
    self.sep_token_id = kwargs.pop("sep_token_id", None)
    #BOS Token (Beginning of Sequence)
    #PAD Token (Padding)
    #EOS Token (End of Sequence)
    #SEP Token (Separator)

    self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

    # task specific arguments
    self.task_specific_params = kwargs.pop("task_specific_params", None)

    # TPU arguments
    self.xla_device = kwargs.pop("xla_device", None)

    # Name or path to the pretrained checkpoint
    self._name_or_path = str(kwargs.pop("name_or_path", ""))

    # Drop the transformers version info
    kwargs.pop("transformers_version", None)

    # Additional attributes without default values
    for key, value in kwargs.items():
      try:
        setattr(self, key, value)
      except AttributeError as err:
        raise err

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
    config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
    return cls.from_dict(config_dict, **kwargs)

  @classmethod
  def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
    with open(json_file, "r", encoding="utf-8") as reader:
      text = reader.read()
    return json.loads(text)

  @classmethod
  def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
    return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

    config = cls(**config_dict)

    if hasattr(config, "pruned_heads"):
      config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())

    # Update config with kwargs if needed
    to_remove = []
    for key, value in kwargs.items():
      if hasattr(config, key):
        setattr(config, key, value)
        to_remove.append(key)
    for key in to_remove:
      kwargs.pop(key, None)

    if return_unused_kwargs:
      return config, kwargs
    else:
      return config

  @classmethod
  def get_config_dict(
    cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
  ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    use_auth_token = kwargs.pop("use_auth_token", None)
    local_files_only = kwargs.pop("local_files_only", False)
    revision = kwargs.pop("revision", None)

    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    if os.path.isdir(pretrained_model_name_or_path):
      config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
    elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
      config_file = pretrained_model_name_or_path
    else:
      config_file = hf_bucket_url(
        pretrained_model_name_or_path, filename=CONFIG_NAME, revision=revision, mirror=None
      )

    try:
      # Load from URL or cache if already cached
      resolved_config_file = cached_path(
        config_file,
        cache_dir=cache_dir,
        force_download=force_download,
        proxies=proxies,
        resume_download=resume_download,
        local_files_only=local_files_only,
        use_auth_token=use_auth_token,
      )
      # Load config dict
      config_dict = cls._dict_from_json_file(resolved_config_file)

    except EnvironmentError as err:
      msg = (
        f"Can't load config for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
        f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
        f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a {CONFIG_NAME} file\n\n"
      )
      raise EnvironmentError(msg)

    except json.JSONDecodeError:
      msg = (
        "Couldn't reach server at '{}' to download configuration file or "
        "configuration file is not a valid JSON file. "
        "Please check network or file content here: {}.".format(config_file, resolved_config_file)
      )
      raise EnvironmentError(msg)

    return config_dict, kwargs

class LlamaConfig(PretrainedConfig):
  model_type = "llama"
  def __init__(
    self,
    vocab_size: int = 32000,
    dim: int = 512,
    dropout: int = 0.0,
    n_layers: int = 8,
    n_heads: int = 8,
    n_kv_heads: Optional[int] = 8,
    max_seq_len: int = 1024,
    layer_norm_eps: float = 1e-5,
    #- 全称 ：Layer Normalization Epsilon
    #作用 ：在层归一化计算中添加的小数值，用于 避免除零错误 和 提高数值稳定性
    multiple_of: int = 32,
    hidden_dim: Optional[int] = None,
    position_embedding_type: str = "rotary",
    #Rotary Position Embedding (RoPE)

    use_cache: bool = True,
    #用于控制模型在 自回归生成过程中是否缓存注意力机制的键值对（key-value pairs
    #- 第一次计算时，存储每一层的键（key）和值（value）张量
    #- 后续生成新 token 时，只计算新 token 的查询（query）
    #- 新 query 与之前缓存的 key/value 进行注意力计算
    #- 时间复杂度： [ o bj ec tO bj ec t ] O ( n ) ，每次生成仅线性增长

    **kwargs
  ):
    super().__init__(**kwargs)

    self.vocab_size = vocab_size
    self.dim = dim
    self.dropout = dropout
    self.n_layers = n_layers
    self.n_heads = n_heads
    self.max_seq_len = max_seq_len
    self.n_kv_heads = n_kv_heads
    self.layer_norm_eps = layer_norm_eps
    self.multiple_of = multiple_of
    self.hidden_dim = hidden_dim
    self.position_embedding_type = position_embedding_type
    self.use_cache = use_cache