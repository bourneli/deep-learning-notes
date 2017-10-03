# -*- coding: utf-8 -*-

import numpy as np
import random
import tensorflow as tf
import time
from src.embedding import RNNEncodeWithTag

print("Current TensorFlow Version", tf.__version__)


##################################################################################
# 参数设置
##################################################################################

tf.flags.DEFINE_string("data_dir", None, "Data file directory")
tf.flags.DEFINE_string("log_dir", None, "Log directory for TensorBoard")
tf.flags.DEFINE_string("data_file", "core_series.csv", "Input Data File")
tf.flags.DEFINE_string("embedding_file", "embedding_features.csv", "Embedding Output File")

tf.flags.DEFINE_integer("train_num", 20000, "Train Data Number")
tf.flags.DEFINE_integer("validate_num", 3000, "Validate Data Number")
tf.flags.DEFINE_integer("class_num", 4, "Tag number")
tf.flags.DEFINE_integer("batch_size", 1000, "Batch Size")
tf.flags.DEFINE_integer("rnn_hidden_num", 24, "RNN hidden Unit Number")
tf.flags.DEFINE_integer("rnn_layer_num", 3, "Number of RNN layers")
tf.flags.DEFINE_integer("run_epoch", 1000, "Training Rounds")

tf.flags.DEFINE_bool("ignore_rank", True, "Ignore the rank feature")
tf.flags.DEFINE_float("learning_rate", 1e-5, "Learning Rate for SGD")
tf.flags.DEFINE_integer("max_series", 60, "Max series length")
tf.flags.DEFINE_integer("validate_span", 100, "Validate Span")
tf.flags.DEFINE_integer("series_feature_num", 3, "Features Number in Series Unit")

conf = tf.flags.FLAGS

if not conf.data_dir:
    raise ValueError("Require --data_dir")

series_input = tf.placeholder(tf.float32, [None, conf.max_series, conf.series_feature_num])
series_length_list = tf.placeholder(tf.int32, [None])
series_target = tf.placeholder(tf.int32, [None, conf.class_num])
ae = RNNEncodeWithTag.RNNEncodeWithTag(
  series_input, series_length_list,
  series_target,
  hidden_num=conf.rnn_hidden_num,
  max_series=conf.max_series,
  learning_rate=conf.learning_rate,
  layer_num=conf.rnn_layer_num,
  activation=tf.nn.tanh)

with tf.name_scope("train_process"):
    summary_loss = tf.summary.scalar("loss", ae.loss)
    summary_error = tf.summary.scalar("error", ae.error)


##################################################################################
# ETL
##################################################################################

# 格式 [dead_norm:hurt_hero_norm:high_level_flag ...]
def decode_series(data_str):
    return [np.array(segment.split(":")[:conf.series_feature_num]).astype(np.float32)
            for segment in data_str.strip().split(" ")]


# tag转vector
def tag_to_array(tag, class_number):
    tag_vector = np.zeros(class_number)
    tag_vector[tag] = 1
    return tag_vector


# 读取原始数据并且解析
with open('%s/%s' % (conf.data_dir, conf.data_file), 'r') as f:
    contents = f.readlines()
raw_data = [line.strip().split(",") for line in contents[1:]]
raw_data = [(row[0],         # id
             int(row[1]),     # tag
             decode_series(row[2]),   # series
             float(row[3]))   # series length
            for row in raw_data]

# 构建训练和预测数据
random.seed(43847584)
random.shuffle(raw_data)

input_raw_data = raw_data[:conf.train_num]
input_data = [np.array(series).reshape([conf.max_series, conf.series_feature_num])
              for [_, _, series, _] in input_raw_data]
input_series_length_list = [int(series_length) for [_, _, _, series_length] in input_raw_data]
input_tag = [tag_to_array(tag, conf.class_num) for [_, tag, _, _] in input_raw_data]

validate_raw_data = raw_data[conf.train_num:(conf.train_num + conf.validate_num)]
validate_id = [round_id for [round_id, _, _, _] in validate_raw_data]
validate_data = [np.array(series).reshape([conf.max_series, conf.series_feature_num])
                 for [_, _, series, _] in validate_raw_data]
validate_series_length_list = [series_length for [_, _, _, series_length] in validate_raw_data]
validate_tag = [tag_to_array(tag, conf.class_num) for [_, tag, _, _] in validate_raw_data]

##################################################################################
# 计算
##################################################################################

sess = tf.Session()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(r'%s/train' % conf.log_dir, sess.graph)
validate_writer = tf.summary.FileWriter('%s/test' % conf.log_dir, sess.graph)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

no_of_batches = int(len(input_data) / conf.batch_size)
start_time = time.clock()
step = 0
for i in range(conf.run_epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp = input_data[ptr:ptr+conf.batch_size]
        inp_length = input_series_length_list[ptr:ptr+conf.batch_size]
        inp_tag = input_tag[ptr:ptr+conf.batch_size]

        ptr += conf.batch_size
        print("Epoch", i, "Ptr", ptr)

        _, current_loss, train_loss, train_error = sess.run([ae.train, ae.loss, summary_loss, summary_error],
                                                  {series_input: inp,
                                                   series_length_list: inp_length,
                                                   series_target: inp_tag})
        # current_loss, train_loss, train_error = sess.run([ae.loss, summary_loss, summary_error],
        #                                                  {series_input: inp,
        #                                                   series_length_list: inp_length,
        #                                                   series_target: inp_tag})

        print(step, current_loss)
        train_writer.add_summary(train_loss, step)
        train_writer.add_summary(train_error, step)

        if step % conf.validate_span == 0:
            [validate_loss, validate_error] = sess.run([summary_loss, summary_error],
                                                       {series_input: validate_data,
                                                        series_length_list: validate_series_length_list,
                                                        series_target: validate_tag})
            validate_writer.add_summary(validate_loss, step)
            validate_writer.add_summary(validate_error, step)

        step += 1
    print("Epoch ", str(i))

end_time = time.clock()
print("Time cost:", end_time - start_time, "Seconds")

# 写结果
[embedding_features, my_soft_pred, my_tag_pred] = sess.run([ae.last_encode_output, ae.softmax_prediction, ae.final_prediction],
                                            {series_input: validate_data,
                                             series_length_list: validate_series_length_list,
                                             series_target: validate_tag})


with open('%s/%s' % (conf.data_dir, conf.embedding_file), 'w') as fw:
    embedding_feature_names = ','.join(['e_%d' % i for i in range(conf.rnn_hidden_num)])
    print(embedding_feature_names)
    fw.write('type,id,%s\n' % embedding_feature_names)
    for i in range(np.shape(embedding_features)[0]):
        (tag, d_id) = validate_id[i].split("-")
        line = '%s,%s,%s\n' % (tag, d_id, ','.join(map(str, embedding_features[i])))
        fw.write(line)

print("Complete")
