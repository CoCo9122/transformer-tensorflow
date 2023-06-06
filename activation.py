import tensorflow as tf
import numpy as np

# GELU関数
def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

# Swish関数
def swish(x):
    return x * tf.math.sigmoid(x)

# Mish関数
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

# 定義通りのSoftmax関数
def softmax_original(x, axis=-1):
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

# Keras式のSoftmax関数
def softmax_stable(x, axis=-1):
    ex = tf.exp(x - tf.reduce_max(x, axis=axis, keepdims=True))
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

# LogSoftmax関数
def logsoftmax(x, axis=-1):
    ex = tf.exp(x)
    return x - tf.math.log(tf.reduce_sum(ex, axis=axis, keepdims=True))