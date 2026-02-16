# Min-Llama Assignment
by Vijay Viswanathan (based on the previous [minbert-assignment](https://github.com/neubig/minbert-assignment))

# 中文注释：项目标题和作者信息
# 这是Min-Llama任务的标题，作者是Vijay Viswanathan，基于之前的minbert-assignment项目

This is an exercise in developing a minimalist version of Llama2, part of Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2024/).

# 中文注释：项目介绍
# 这是一个开发Llama2极简版本的练习，是卡内基梅隆大学CS11-711高级NLP课程的一部分

In this assignment, you will implement some important components of the Llama2 model to better understanding its architecture. 
You will then perform sentence classification on ``sst`` dataset and ``cfimdb`` dataset with this model.

# 中文注释：任务目标
# 在这个作业中，你将实现Llama2模型的一些重要组件，以更好地理解其架构。
# 然后，你将使用这个模型在sst数据集和cfimdb数据集上执行句子分类任务.

## Assignment Details

# 中文注释：作业详情
# 这部分详细介绍了作业的具体内容和要求

### Your task
# 中文注释：你的任务
# 这部分描述了你需要完成的具体任务
The code to implement can be found in `llama.py`, `classifier.py` and `optimizer.py`. You are reponsible for writing _core components_ of Llama2 (one of the leading open source language models). In doing so, you will gain a strong understanding of neural language modeling. We will load pretrained weights for your language model from `stories42M.pt`; an 8-layer, 42M parameter language model pretrained on the [TinyStories](https://arxiv.org/abs/2305.07759) dataset (a dataset of machine-generated children's stories). This model is small enough that it can be trained (slowly) without a GPU. You are encouraged to use Colab or a personal GPU machine (e.g. a Macbook) to be able to iterate more quickly.

# 中文注释：实现代码位置和模型信息
# 你需要在llama.py、classifier.py和optimizer.py文件中实现代码。你负责编写Llama2（领先的开源语言模型之一）的核心组件。
# 通过这样做，你将获得对神经语言建模的深入理解。我们将从stories42M.pt加载预训练权重，这是一个8层、42M参数的语言模型，在TinyStories数据集（机器生成的儿童故事数据集）上预训练。
# 这个模型足够小，可以在没有GPU的情况下训练（速度较慢）。建议你使用Colab或个人GPU机器（如Macbook）来更快地迭代。

Once you have implemented these components, you will test our your model in 3 settings:
1) Generate a text completion (starting with the sentence `"I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is"`). You should see coherent, grammatical English being generated (though the content and topicality of the completion may be absurd, since this LM was pretrained exclusively on children's stories).
2) Perform zero-shot, prompt-based sentiment analysis on two datasets (SST-5 and CFIMDB). This will give bad results (roughly equal to choosing a random target class).
3) Perform task-specific finetuning of your Llama2 model, after implementing a classification head in `classifier.py`. This will give much stronger classification results.
4) If you've done #1-3 well, you will get an A! However, since you've come this far, try implementing something new on top of your hand-written language modeling system! If your method provides strong empirical improvements or demonstrates exceptional creativity, you'll get an A+ on this assignment.

# 中文注释：模型测试设置
# 一旦实现了这些组件，你将在3个设置中测试你的模型：
# 1) 生成文本补全（以句子"I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is"开头）。你应该看到连贯、语法正确的英语生成（尽管内容和主题可能很荒谬，因为这个LM专门在儿童故事上预训练）。
# 2) 在两个数据集（SST-5和CFIMDB）上执行零样本、基于提示的情感分析。这会得到不好的结果（大致相当于选择随机目标类别）。
# 3) 在classifier.py中实现分类头后，对Llama2模型进行任务特定的微调。这将给出更强的分类结果。
# 4) 如果你做好了#1-3，你会得到A！然而，既然你已经走了这么远，尝试在你手写的语言建模系统之上实现一些新东西！如果你的方法提供了强有力的实证改进或展示了卓越的创造力，你将在这个作业中获得A+。

### Important Notes
# 中文注释：重要注意事项
# 这部分列出了完成作业时需要注意的重要事项
* Follow `setup.sh` to properly setup the environment and install dependencies.
* There is a detailed description of the code structure in [structure.md](./structure.md), including a description of which parts you will need to implement.
* You are only allowed to use libraries that are installed by `setup.sh`, no other external libraries are allowed (e.g., `transformers`).
* The `data/cfimdb-test.txt` file provided to you does **not** contain gold-labels, and contains a placeholder negative (-1) label. Evaluating your code against this set will show lower accuracies so do not worry if the numbers don't make sense.
* We will run your code with commands below (under "Reference outputs/accuracies"), so make sure that whatever your best results are reproducible using these commands.
    * Do not change any of the existing command options (including defaults) or add any new required parameters

# 中文注释：重要注意事项详情
# * 按照setup.sh正确设置环境并安装依赖项。
# * 在structure.md中有代码结构的详细描述，包括你需要实现哪些部分的描述。
# * 你只允许使用setup.sh安装的库，不允许使用其他外部库（例如transformers）。
# * 提供给你的data/cfimdb-test.txt文件不包含黄金标签，而是包含一个占位符负（-1）标签。根据这个集合评估你的代码会显示较低的准确率，所以如果数字没有意义，不要担心。
# * 我们将使用下面（在"Reference outputs/accuracies"下）的命令运行你的代码，所以确保你的最佳结果可以使用这些命令重现。
#     * 不要更改任何现有的命令选项（包括默认值）或添加任何新的必需参数。

## Reference outputs/accuracies: 
# 中文注释：参考输出/准确率
# 这部分提供了模型在不同任务上的参考输出和准确率

*Text Continuation* (`python run_llama.py --option generate`)
You should see continuations of the sentence `I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is...`. We will generate two continuations - one with temperature 0.0 (which should have a reasonably coherent, if unusual, completion) and one with temperature 1.0 (which is likely to be logically inconsistent and may contain some coherence or grammar errors).

# 中文注释：文本续接
# 使用命令python run_llama.py --option generate运行文本续接任务。
# 你应该看到句子"I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is..."的续接。
# 我们将生成两个续接 - 一个温度为0.0（应该有合理连贯的完成，即使不寻常），一个温度为1.0（可能逻辑不一致，可能包含一些连贯性或语法错误）。

*Zero Shot Prompting*
Zero-Shot Prompting for SST:

`python run_llama.py --option prompt --batch_size 10  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-prompting-output.txt --test_out sst-test-prompting-output.txt [--use_gpu]`
or 
`python run_llama.py --option prompt --batch_size 64  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-prompting-output.txt --test_out sst-test-prompting-output.txt --use_gpu` 

Prompting for SST:
Dev Accuracy: 0.213 (0.000)
Test Accuracy: 0.224 (0.000)

Zero-Shot Prompting for CFIMDB:

`python run_llama.py --option prompt --batch_size 10  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-prompting-output.txt --test_out cfimdb-test-prompting-output.txt [--use_gpu]`

Prompting for CFIMDB:
Dev Accuracy: 0.498 (0.000)
Test Accuracy: -

# 中文注释：零样本提示
# 这部分展示了零样本提示在SST和CFIMDB数据集上的性能。
# SST数据集的零样本提示开发集准确率为0.213，测试集准确率为0.224。
# CFIMDB数据集的零样本提示开发集准确率为0.498，测试集没有提供准确率（因为测试集没有黄金标签）。

*Classification Finetuning*

`python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 80  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-finetuning-output.txt --test_out sst-test-finetuning-output.txt [--use_gpu]`

Finetuning for SST:
Dev Accuracy: 0.414 (0.014)
Test Accuracy: 0.418 (0.017)

`python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 10  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-finetuning-output.txt --test_out cfimdb-test-finetuning-output.txt [--use_gpu]`

Finetuning for CFIMDB:
Dev Accuracy: 0.800 (0.115)
Test Accuracy: -

Mean reference accuracies over 10 random seeds with their standard deviation shown in brackets.

# 中文注释：分类微调
# 这部分展示了分类微调在SST和CFIMDB数据集上的性能。
# SST数据集的微调开发集准确率为0.414（标准差0.014），测试集准确率为0.418（标准差0.017）。
# CFIMDB数据集的微调开发集准确率为0.800（标准差0.115），测试集没有提供准确率。
# 所有参考准确率都是在10个随机种子上的平均值，括号中显示标准差。

### Submission
# 中文注释：提交要求
# 这部分详细说明了作业的提交要求和格式

**Code:**
You will submit a full code package, with output files, on **Canvas**. This package will be checked by the TAs in the 1-2 weeks 
   after the assignment for its correctness and executability.

**Report (optional):** Your zip file can include a pdf file, named ANDREWID-report.pdf, if (1) you've implemented something else on top of the requirements and further improved accuracy for possible extra points (see "Grading" below), and/or (2) if your best results are with some hyperparameters other than the default, and you want to specify how we should run your code. If you're doing (1), we expect your report should be 1-2 pages, but no more than 3 pages. If you're doing (2), the report can be very brief.

# 中文注释：代码和报告提交
# **代码：**
# 你将在Canvas上提交一个完整的代码包，包括输出文件。这个包将在作业提交后的1-2周内由助教检查其正确性和可执行性。
# 
# **报告（可选）：** 如果(1)你在要求之外实现了其他功能并进一步提高了准确率以获得可能的额外分数（见下面的"评分"），和/或(2)如果你的最佳结果是使用默认以外的一些超参数，并且你想指定我们应该如何运行你的代码，你的zip文件可以包含一个名为ANDREWID-report.pdf的pdf文件。如果你在做(1)，我们希望你的报告应该是1-2页，但不超过3页。如果你在做(2)，报告可以非常简短。

#### Canvas Submission

For submission via [Canvas](https://canvas.cmu.edu/),
the submission file should be a zip file with the following structure (assuming the
lowercase Andrew ID is ``ANDREWID``):
```
ANDREWID/
├── run_llama.py
├── base_llama.py
├── llama.py
├── rope.py
├── classifier.py
├── config.py
├── optimizer.py
├── sanity_check.py
├── tokenizer.py
├── utils.py
├── README.md
├── structure.md
├── sanity_check.data
├── generated-sentence-temp-0.txt
├── generated-sentence-temp-1.txt
├── [OPTIONAL] sst-dev-advanced-output.txt
├── [OPTIONAL] sst-test-advanced-output.txt
├── sst-dev-prompting-output.txt
├── sst-test-prompting-output.txt
├── sst-dev-finetuning-output.txt
├── sst-test-finetuning-output.txt
├── [OPTIONAL] cfimdb-dev-advanced-output.txt
├── [OPTIONAL] cfimdb-test-advanced-output.txt
├── cfimdb-dev-prompting-output.txt
├── cfimdb-test-prompting-output.txt
├── cfimdb-dev-finetuning-output.txt
├── cfimdb-test-finetuning-output.txt
└── setup.sh
```

`prepare_submit.py` can help to create(1) or check(2) the to-be-submitted zip file. It
will throw assertion errors if the format is not expected, and *submissions that fail
this check will be graded down*.

Usage:
1. To create and check a zip file with your outputs, run
   `python3 prepare_submit.py path/to/your/output/dir ANDREWID`
2. To check your zip file, run
   `python3 prepare_submit.py path/to/your/submit/zip/file.zip ANDREWID`

Please double check this before you submit to Canvas; most recently we had about 10/100
students lose a 1/3 letter grade because of an improper submission format.

# 中文注释：Canvas提交
# 通过Canvas提交时，提交文件应该是一个zip文件，具有以下结构（假设小写的Andrew ID是ANDREWID）：
# 上面列出了详细的文件结构，包括所有需要提交的文件。
# 
# prepare_submit.py可以帮助创建(1)或检查(2)要提交的zip文件。如果格式不符合预期，它会抛出断言错误，并且*未通过此检查的提交将被降级*。
# 
# 使用方法：
# 1. 要创建并检查带有输出的zip文件，运行 python3 prepare_submit.py path/to/your/output/dir ANDREWID
# 2. 要检查你的zip文件，运行 python3 prepare_submit.py path/to/your/submit/zip/file.zip ANDREWID
# 
# 请在提交到Canvas之前仔细检查；最近我们有大约10/100的学生因为提交格式不正确而失去了1/3的字母等级。


### Grading
# 中文注释：评分标准
# 这部分详细说明了作业的评分标准，包括不同等级的要求
* A+: (Advanced implementation) You additionally implement something else on top of the requirements for A, and achieve significant accuracy improvements or demonstrate exceptional creativity. This improvement can be in either the zero-shot setting (no task-specific finetuning required) or in the finetuning setting (improving over our current finetuning implementation). Please write down the things you implemented and experiments you performed in the report. You are also welcome to provide additional materials such as commands to run your code in a script and training logs.
    * perform [continued pre-training](https://arxiv.org/abs/2004.10964) using the language modeling objective to do domain adaptation
    * enable zero-shot prompting using a more principled inference algorithm than our current implementation. For example, we did not include an attention mask despite right-padding all inputs (to enable batch prediction); this could be improved.
    * perform [prompt-based finetuning](https://arxiv.org/abs/2109.01247)
    * add [regularization](https://arxiv.org/abs/1909.11299) to our finetuning process
    * try parameter-efficient finetuning (see Section 2.2 [here](https://arxiv.org/abs/2110.04366) for an overview)
    * try alternative fine-tuning algorithms e.g. [SMART](https://www.aclweb.org/anthology/2020.acl-main.197) or [WiSE-FT](https://arxiv.org/abs/2109.01903)
    * add other model components on top of the model
* A: You implement all the missing pieces and the original ``classifier.py`` with ``--option prompt`` and ``--option finetune`` code such that coherent text (i.e. mostly grammatically well-formed) can be generated and the model achieves comparable accuracy (within 0.05 accuracy for SST or 0.15 accuracy for CFIMDB) to our reference implementation.
* A-: You implement all the missing pieces and the original ``classifier.py`` with ``--option prompt`` and ``--option finetune`` code but coherent text is not generated (i.e. generated text is not well-formed English) or accuracy is not comparable to the reference (accuracy is more than 0.05 accuracy or 0.15 accuracy from our reference scores, for for SST and CFIMDB, respectively).
* B+: All missing pieces are implemented and pass tests in ``sanity_check.py`` (llama implementation) and ``optimizer_test.py`` (optimizer implementation)
* B or below: Some parts of the missing pieces are not implemented.

# 中文注释：评分等级说明
# * A+：（高级实现）你在A的要求基础上额外实现了其他功能，并取得了显著的准确率改进或展示了卓越的创造力。这种改进可以在零样本设置（不需要任务特定的微调）或微调设置（改进我们当前的微调实现）中。请在报告中写下你实现的内容和进行的实验。你也可以提供额外的材料，如运行代码的命令脚本和训练日志。
#   * 使用语言建模目标进行[持续预训练](https://arxiv.org/abs/2004.10964)以进行领域适应
#   * 使用比我们当前实现更有原则的推理算法启用零样本提示。例如，尽管我们对所有输入进行了右填充（以启用批量预测），但我们没有包括注意力掩码；这可以改进。
#   * 执行[基于提示的微调](https://arxiv.org/abs/2109.01247)
#   * 在我们的微调过程中添加[正则化](https://arxiv.org/abs/1909.11299)
#   * 尝试参数高效的微调（参见[这里](https://arxiv.org/abs/2110.04366)的第2.2节概述）
#   * 尝试替代微调算法，例如[SMART](https://www.aclweb.org/anthology/2020.acl-main.197)或[WiSE-FT](https://arxiv.org/abs/2109.01903)
#   * 在模型之上添加其他模型组件
# * A：你实现了所有缺失的部分和原始的classifier.py，带有--option prompt和--option finetune代码，使得可以生成连贯的文本（即大多数字法良好），并且模型达到与我们的参考实现相当的准确率（SST的准确率在0.05以内，CFIMDB的准确率在0.15以内）。
# * A-：你实现了所有缺失的部分和原始的classifier.py，带有--option prompt和--option finetune代码，但没有生成连贯的文本（即生成的文本不是良好的英语）或准确率与参考不相当（SST和CFIMDB的准确率分别与我们的参考分数相差超过0.05或0.15）。
# * B+：所有缺失的部分都已实现，并通过了sanity_check.py（llama实现）和optimizer_test.py（optimizer实现）中的测试
# * B或更低：部分缺失的部分未实现。

If your results can be confirmed through the submitted files, but there are problems with your
code submitted through Canvas, such as not being properly formatted, not executing in
the appropriate amount of time, etc., you will be graded down 1/3 grade (e.g. A+ -> A or A- -> B+).

All assignments must be done individually and we will be running plagiarism detection
on your code. If we confirm that any code was plagiarized from that of other students
in the class, you will be subject to strict measure according to CMUs academic integrity
policy. That being said, *you are free to use publicly available resources* (e.g. papers or open-source
code), but you ***must provide proper attribution***.

# 中文注释：其他评分说明
# 如果你的结果可以通过提交的文件确认，但你通过Canvas提交的代码存在问题，例如格式不正确、未在适当的时间内执行等，你的评分将降低1/3等级（例如A+ -> A或A- -> B+）。
# 
# 所有作业必须单独完成，我们将对你的代码进行 plagiarism 检测。如果我们确认任何代码是从班上其他学生那里抄袭的，你将根据CMU的学术诚信政策受到严格的措施。也就是说，*你可以自由使用公开可用的资源*（例如论文或开源代码），但你***必须提供适当的归因***。

### Acknowledgement
# 中文注释：致谢
# 这部分感谢了代码的来源
This code is based on llama2.c by Andrej Karpathy. Parts of the code are also from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

# 中文注释：代码来源
# 此代码基于Andrej Karpathy的llama2.c。部分代码也来自[transformers](https://github.com/huggingface/transformers)库（[Apache License 2.0](./LICENSE)）。
