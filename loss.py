import tensorflow as tf
from activation import logsoftmax

def crossentropy(labels, logits):
    # バッチサイズ×トークン列を一次元に並べる
    num_vocabrary = logits.shape.as_list()[-1]
    flat_labels = tf.reshape(labels, [-1])
    flat_labels = tf.cast(flat_labels, tf.int32)
    flat_logits = tf.reshape(logits, [-1, num_vocabrary])
    # one-hotにする
    one_hot_labels = tf.one_hot(flat_labels, depth=num_vocabrary, dtype=tf.float32)
    # Softmaxの対数の負がクロスエントロピー
    log_prods = logsoftmax(flat_logits)
    # 該当箇所の損失を求める
    loss = -tf.reduce_sum(log_prods * one_hot_labels, axis=[-1])
    # 逆伝搬する損失を求める
    loss = tf.reduce_mean(loss) # 損失の平均値
    return loss