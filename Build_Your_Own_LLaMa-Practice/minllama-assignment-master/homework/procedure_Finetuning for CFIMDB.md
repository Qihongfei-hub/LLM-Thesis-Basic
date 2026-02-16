### ?
The model was using padding token hidden states for classification




###
python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 10  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-finetuning-output.txt --test_out cfimdb-test-finetuning-output.txt --use_gpu


python run_llama.py --option finetune --epochs 1 --lr 2e-5 --batch_size 10  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-finetuning-output.txt --test_out cfimdb-test-finetuning-output.txt --use_gpu

### 1 epoch
args: {'train': 'data/cfimdb-train.txt', 'dev': 'data/cfimdb-dev.txt', 'test': 'data/cfimdb-test.txt', 'label_names': 'data/cfimdb-label-mapping.json', 'pretrained_model_path': 'stories42M.pt', 'max_sentence_len': None, 'seed': 1337, 'epochs': 1, 'option': 'finetune', 'use_gpu': True, 'generated_sentence_low_temp_out': 'generated-sentence-temp-0.txt', 'generated_sentence_high_temp_out': 'generated-sentence-temp-1.txt', 'dev_out': 'cfimdb-dev-finetuning-output.txt', 'test_out': 'cfimdb-test-finetuning-output.txt', 'batch_size': 10, 'hidden_dropout_prob': 0.3, 'lr': 2e-05}
load 1707 data from data/cfimdb-train.txt
load 245 data from data/cfimdb-dev.txt
train-0: 100%|████████████████████████████████████████████████████| 171/171 [05:19<00:00,  1.87s/it]  
eval: 100%|████████████████████████████████████████████████████████| 171/171 [01:57<00:00,  1.46it/s] 
eval: 100%|██████████████████████████████████████████████████████████| 25/25 [00:17<00:00,  1.43it/s] 
save the model to finetune-1-2e-05.pt████████████████████████████████| 25/25 [00:17<00:00,  1.51it/s] 
epoch 0: train loss :: 0.936, train acc :: 0.504, dev acc :: 0.502
100%|████████████████████████████████████████████████████████████████| 1/1 [07:34<00:00, 454.27s/it]  
load model from finetune-1-2e-05.pt
load 245 data from data/cfimdb-dev.txt
load 488 data from data/cfimdb-test.txt
eval: 100%|██████████████████████████████████████████████████████████| 25/25 [00:16<00:00,  1.53it/s] 
eval: 100%|██████████████████████████████████████████████████████████| 49/49 [00:30<00:00,  1.59it/s] 
dev acc :: 0.502
test acc :: 1.000
(llama_hw) PS C:\Users\hongf\LLM-Thesis-Basic\Build_Your_Own_LLaMa-Practice\minllama-assignment-master> 


- The user runs``python run_llama.py --option finetune --epochs 1 --lr 2e-5 --batch_size 10 --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-finetuning-output.txt --test_out cfimdb-test-finetuning-output.txt --use_gpu``
- In the code:

- ``get_args()`` parses the command-line arguments
- ``args.filepath`` is set to``finetune-1-2e-05.pt``
- ``seed_everything()`` fixes the random seed
- Since``args.option == "finetune"`` , it calls``train(args)``
- In``train(args)`` :

- Loads the train and dev data using``create_data()``
- Creates a``LlamaEmbeddingClassifier`` model
- The model loads a pre-trained Llama model from``stories42M.pt``
- All Llama parameters are set to``requires_grad = True`` for fine-tuning
- A classification head is added on top of the Llama model
- The model is trained for 1 epoch
- After training, it calls``test(args)``
- In``test(args)`` :

- Loads the saved model from``finetune-1-2e-05.pt``
- Evaluates it on dev and test sets
- Writes predictions to output files
- The issue: Looking at the output, we see:

epoch 0: train loss :: 0.936, train acc :: 0.504, dev acc :: 0.502
...
dev acc :: 0.502
test acc :: 1.000