# doc-grounded-dialog-with-transformers

This repository provides the Pytorch source code for Transformer-based span-selection for goal-oriented information-seeking dialogue systems. In particular, the goal is to teach a dialogue system to identify the most relevant knowledge in the associated document for generating agent responses in natural language. Details are described in 
[doc2dial: A Goal-Oriented Document-Grounded Dialogue Dataset](https://www.aclweb.org/anthology/2020.emnlp-main.652/), in EMNLP 2020.  

Built with transformers from HuggingFace🤗. (Thanks HuggingFace!)
 
### Dataset
The [Doc2Dial dataset](https://doc2dial.github.io/workshop2021/file/doc2dial_sharedtask.zip) contains goal-oriented conversations between an end user and an assistive agent. Each turn in a conversation is annotated with a dialogue scene, which includes speaker role, dialogue act, and grounding in a document.
Detailed description can be found at the dataset webpage [here](https://doc2dial.github.io/workshop2021/data_readme.html).

### Citation
Please use the bibtex entry below to cite our [paper](https://www.aclweb.org/anthology/2020.emnlp-main.652/) if you use the dataset or the baseline code. Thank you!
```bibtex
@inproceedings{feng-etal-2020-doc2dial,
    title = "doc2dial: A Goal-Oriented Document-Grounded Dialogue Dataset",
    author = "Feng, Song  and Wan, Hui  and Gunasekara, Chulaka  and Patel, Siva  and Joshi, Sachindra  and Lastras, Luis",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.652",
}
```

### Requirements
 * Python (3.7 or higher)
 * Pytorch (1.1.0 or higher)
 * HuggingFace / Transformer (3.3.0)
 * Numpy
 * tensorboard
 * tqdm

These can be installed using pip by running :
```bash
pip install -r requirements.txt
```

### Finetuning 
Example command to train a model on Doc2Dial training set (with distributed training on 2 V100 GPUs):
```bash
python -m torch.distributed.launch --nproc_per_node=2 knowledge_identification/run_doc2dial.py 
    --model_type bert 
    --model_name_or_path bert-base-uncased
    --do_train 
    --evaluate_during_training
    --logging_steps 500
    --save_steps 500
    --do_lower_case 
    --data_dir $datadir
    --doc_file doc2dial_doc_data.json 
    --train_dial_file doc2dial_dial_data_train.json 
    --eval_dial_file doc2dial_dial_data_dev.json 
    --learning_rate 3e-5 
    --num_train_epochs 3
    --max_seq_length 512 
    --doc_stride 128
    --output_dir $output_dir
    --overwrite_output_dir 
    --per_gpu_eval_batch_size=27   
    --per_gpu_train_batch_size=27   
    --gradient_accumulation_steps=1 
    --warmup_steps=500
    --weight_decay=0.01
    --fp16
    --get_utterances all
    --max_query_length 256
```

### Prediction 
Example command to decode Doc2Dial test set with a model trained with the above setting:
```bash
python  $codedir/knowledge_identification/run_doc2dial.py 
    --model_type bert 
    --model_name_or_path $model_dir 
    --do_eval 
    --do_lower_case 
    --data_dir $datadir
    --doc_file doc2dial_doc_data.json 
    --eval_dial_file doc2dial_dial_data_test.json 
    --max_seq_length 512 
    --doc_stride 128
    --output_dir $output_dir
    --overwrite_output_dir 
    --per_gpu_eval_batch_size=27 
    --fp16
    --get_utterances all
    --max_query_length 256
```



