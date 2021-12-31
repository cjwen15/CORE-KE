### PLM fine-tuning

#### Step1. fine-tune the PLM on external knowledge corpus:

```shell
python bert/create_pretraining_data.py \
  --input_file= data/fine-tune_text_wiki.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

```shell
python bert/run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=/tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
```

you can modify the parameters, reference: https://github.com/google-research/bert#pre-training-with-bert

Then take the output from 'pretraining_output' for the next step.


#### Step2. fine-tune the model on DSNFs extraction:
(code for DSNFs extraction, reference: https://github.com/lemonhu/open-entity-relation-extraction)

First change the 'bert_path' in 'bert_lstm_ner.py' with the output model of **step 1**,

then run the code:

```shell
python bert_lstm_ner.py
```

* Take the output model from **step 2** for the Chinese ORE in 'main.py'.
