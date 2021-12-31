# @File  : BERT_COIE_MaskedCRF.py
# --*--coding: utf-8 --*--
# @Author: Wen
# @Date  : 2021/7/12

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bert import modeling
from bert import optimization
from bert import tokenization

import tensorflow as tf
import numpy as np
from utils import tf_metrics
import pickle
import collections
import os
import json
import codecs

from lstm_crf_layer_mask import BLSTM_CRF
from tensorflow.contrib.layers.python.layers import initializers

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", 'data/',
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file", 'bert_model/chinese_wobert_plus_L-12_H-768_A-12/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", "ORE", "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir", 'output/',
    "The output directory where the full model checkpoints will be written."
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", 'bert_model/chinese_wobert_plus_L-12_H-768_A-12/bert_model.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", True, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("pos_embedding_size", 5, "The size of pos_embedding for one feature")

flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 16, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 16, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")

# 0518
flags.DEFINE_string('data_config_path', './data.conf',
                    'data config file, which save train and dev config')

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", 'bert_model/chinese_wobert_plus_L-12_H-768_A-12/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer('lstm_size', 128, 'size of lstm units')
flags.DEFINE_integer('num_layers', 1, 'number of rnn layers, default is 1')
flags.DEFINE_string('cell', 'lstm', 'which rnn cell used')


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, pos_embedding, dp_embedding, head_embedding, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.pos_embedding = pos_embedding
        self.dp_embedding = dp_embedding
        self.head_embedding = head_embedding


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, pos_embedding, dp_embedding):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.pos_embedding = pos_embedding
        self.dp_embedding = dp_embedding


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with open(input_file, encoding='utf-8') as f:
            lines = []

            for line in f:
                line = json.loads(line)
                words = ' '.join(list(line['sentence']))
                labels = ''.join(line['label'])
                poss = line['pos_tag']
                dps = line['dp_tag']
                head = line['head_tag']
                lines.append([labels, words, poss, dps, head])

            return lines


class OreProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "COER/train_213327_add_feature.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "COER/test_2000_add_feature.txt")), "test"
		)

    def get_labels(self):
        return ["O", "B-E1", "I-E1", "B-E2", "I-E2", "B-R", "I-R", "X", "[CLS]", "[SEP]"]

    def valid_labels(self):
        return [1, 2, 3, 4, 5, 6]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            pos_embedding = line[2]
            dp_embedding = line[3]
            head_embedding = line[4]
            examples.append(
                InputExample(guid=guid, text=text, label=label, dp_embedding=dp_embedding, head_embedding=head_embedding, pos_embedding=pos_embedding))
        return examples


def write_tokens(tokens, mode):
    if mode == "test":
        path = os.path.join(FLAGS.output_dir, "token_" + mode + ".txt")
        wf = open(path, 'a')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')
        wf.close()


def convert_single_example(ex_index, example, pos_mat, dp_mat, head_mat, label_list, max_seq_length, tokenizer, mode):

    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    with open('./output/label2id.pkl', 'wb') as w:
        pickle.dump(label_map, w)
    origin_path = os.path.dirname(os.path.abspath(__file__))

    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        if labels[i] in label_map:
            label_ids.append(label_map[labels[i]])
        else:
            label_ids.append(1)
    ntokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("**NULL**")

    pos_embedding = np.zeros((max_seq_length, 10), dtype=np.float)
    for i, p in enumerate(example.pos_embedding):
        if i == max_seq_length:
            break
        pos_embedding[i] = pos_mat[p]

    dp_embedding = np.zeros((max_seq_length, 15), dtype=np.float)
    for i, p in enumerate(example.dp_embedding):
        if i == max_seq_length:
            break
        h = example.head_embedding[i]
        if h < max_seq_length:
            dp_embedding[i] = np.concatenate((dp_mat[p], head_mat[h]), axis=-1)
        else:
            pad = np.random.uniform(0, 10, (5,))
            dp_embedding[i] = np.concatenate((dp_mat[p], pad), axis=-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert pos_embedding.shape == (max_seq_length, 10)
    assert dp_embedding.shape == (max_seq_length, 15)

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        pos_embedding=pos_embedding,
        dp_embedding=dp_embedding
    )
    write_tokens(ntokens, mode)
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, mode=None
):

    origin_path = os.path.dirname(os.path.abspath(__file__))

    posfile = os.path.join(origin_path, "data/COER/pos_mat.npy")
    dptfile = os.path.join(origin_path, "data/COER/dp_mat.npy")
    headfile = os.path.join(origin_path, "data/COER/head_mat.npy")
    pos_mat = np.load(posfile)
    dp_mat = np.load(dptfile)
    head_mat = np.load(headfile)

    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        feature = convert_single_example(ex_index, example, pos_mat, dp_mat, head_mat, label_list, max_seq_length,
                                         tokenizer, mode)
        pos_feature = feature.pos_embedding.flatten()
        dp_feature = feature.dp_embedding.flatten()

        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["pos_embedding"] = tf.train.Feature(float_list=tf.train.FloatList(value=pos_feature))
        features["dp_embedding"] = tf.train.Feature(float_list=tf.train.FloatList(value=dp_feature))
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

    if mode == 'test':
        batch_size = FLAGS.predict_batch_size
        if not len(examples) % batch_size == 0:
            input_ids = [0] * max_seq_length
            input_mask = [0] * max_seq_length
            segment_ids = [0] * max_seq_length
            label_ids = [0] * max_seq_length
            pos_embedding = np.zeros((max_seq_length, 10), dtype=np.float)
            dp_embedding = np.zeros((max_seq_length, 15), dtype=np.float)
            feature = InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
                pos_embedding=pos_embedding,
                dp_embedding=dp_embedding
            )
            for i in range(batch_size - len(examples) % batch_size):
                features = collections.OrderedDict()
                features["input_ids"] = create_int_feature(feature.input_ids)
                features["input_mask"] = create_int_feature(feature.input_mask)
                features["segment_ids"] = create_int_feature(feature.segment_ids)
                features["label_ids"] = create_int_feature(feature.label_ids)
                features["pos_embedding"] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=pos_feature))
                features["dp_embedding"] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=dp_feature))
                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_example.SerializeToString())

def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "pos_embedding": tf.VarLenFeature(tf.float32),
        "dp_embedding": tf.VarLenFeature(tf.float32)
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)

        pos_embedding = tf.sparse_tensor_to_dense(example['pos_embedding'], default_value=0)
        pos_embedding = tf.reshape(pos_embedding, [seq_length, 10])
        example['pos_embedding'] = pos_embedding

        dp_embedding = tf.sparse_tensor_to_dense(example['dp_embedding'], default_value=0)
        dp_embedding = tf.reshape(dp_embedding, [seq_length, 15])
        example['dp_embedding'] = dp_embedding

        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, pos_embedding, dp_embedding, num_labels,
                 use_one_hot_embeddings, label2idx_map):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value
    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)
    blstm_crf = BLSTM_CRF(embedded_chars=embedding, hidden_unit=FLAGS.lstm_size, cell_type=FLAGS.cell,
                          num_layers=FLAGS.num_layers,
                          droupout_rate=0.9, initializers=initializers, num_labels=num_labels,
                          seq_length=FLAGS.max_seq_length, labels=labels, lengths=lengths, is_training=is_training,
                          label2idx_map=label2idx_map)
    rst = blstm_crf.add_blstm_crf_layer()
    return rst

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, valid_labels):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        pos_embedding = features["pos_embedding"]
        dp_embedding = features["dp_embedding"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        label2id_map = {}
        for idx, label in enumerate(["O", "B-E1", "I-E1", "B-E2", "I-E2", "B-R", "I-R", "X", "[CLS]", "[SEP]"]):
            label2id_map[label] = idx

        (total_loss, logits, _, predicts) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids, pos_embedding, dp_embedding,
            num_labels, use_one_hot_embeddings, label2id_map)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(label_ids, predicts, valid_labels):
                precision = tf_metrics.precision(label_ids, predicts, num_labels, valid_labels,
                                                 average="macro")
                recall = tf_metrics.recall(label_ids, predicts, num_labels, valid_labels,
                                           average="macro")
                f = tf_metrics.f1(label_ids, predicts, num_labels, valid_labels, average="macro")
                return {
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f": f,
                }

            eval_metrics = (metric_fn, [label_ids, predicts, valid_labels])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predicts, scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "ore": OreProcessor
    }
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels()
    valid_label = processor.valid_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if os.path.exists(FLAGS.data_config_path):
        with codecs.open(FLAGS.data_config_path) as fd:
            data_config = json.load(fd)
    else:
        data_config = {}

    if FLAGS.do_train:
        if len(data_config) == 0:
            train_examples = processor.get_train_examples(FLAGS.data_dir)
            num_train_steps = int(
                len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
            num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

            data_config['num_train_steps'] = num_train_steps
            data_config['num_warmup_steps'] = num_warmup_steps
            data_config['num_train_size'] = len(train_examples)
        else:
            num_train_steps = int(data_config['num_train_steps'])
            num_warmup_steps = int(data_config['num_warmup_steps'])


    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        valid_labels=valid_label
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")

        filed_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, 'train')
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")

        filed_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file, 'eval')

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        eval_steps = None
        if FLAGS.use_tpu:
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        eval_drop_remainder = True
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        result = estimator.evaluate(input_fn=eval_input_fn)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")

    if FLAGS.do_predict:
        token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
        with open('./output/label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
        if os.path.exists(token_path):
            os.remove(token_path)
        predict_examples = processor.get_test_examples(FLAGS.data_dir)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")

        filed_based_convert_examples_to_features(
            predict_examples, label_list, FLAGS.max_seq_length, tokenizer, predict_file, 'test')

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")
        predict_drop_remainder = False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        predicted_result = estimator.evaluate(input_fn=predict_input_fn)
        output_eval_file = os.path.join(FLAGS.output_dir, "predicted_results.txt")
        with codecs.open(output_eval_file, "w", encoding='utf-8') as writer:
            tf.logging.info("***** Predict results *****")
            for key in sorted(predicted_result.keys()):
                tf.logging.info("  %s = %s", key, str(predicted_result[key]))
                writer.write("%s = %s\n" % (key, str(predicted_result[key])))

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")

        with open(output_predict_file, 'w') as writer:
            for prediction in result:
                output_line = " ".join(id2label[id] for id in prediction if id != 0) + "\n"
                writer.write(output_line)


if __name__ == "__main__":
    '''
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    '''
    tf.app.run()



