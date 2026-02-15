conda activate my_project_env
pip freeze > requirements.txts
pip install -r requirements.txt


python run_llama.py --option generate

FileNotFoundError: [Errno 2] No such file or directory: 'stories42M.pt'








python --version
Python 3.11.14

https://pypi.tuna.tsinghua.edu.cn/simple

!pip install 'ms-swift[all]' -U
# ModelScope 镜像需要补充安装依赖 tf keras
!pip install tf_keras
# 让 swift 可以被作为全局 CLI 使用
!pip install 'ms-swift'
# 以及相关训练和推理验证依赖 transfomers（peft）,vllm ( ModelScope 环境模型带，需要升级下）
!pip install transformers vllm -U

mkdir ./LLM-Research
git clone https://www.modelscope.cn/LLM-Research/Meta-Llama-3.1-8B-Instruct.git ./LLM-Research/

https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct

git clone https://hf-mirror.com/datasets/kigner/ruozhiba-llama3


https://ollama.com/ticlazau/meta-llama-3.1-8b-instruct

