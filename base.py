import tensorflow as tf

# 全結合層
class FC(tf.keras.layers.Layer):
    def __init__(self, n_inputs, n_feats, name, w_init_stdev=0.02):
     super(FC, self).__init__(name=name)
        nx = n_inputs
        nf = n_feats
        # 重み
        wb = tf.random.normal([nx, nf], stddev=w_init_stdev)
        self.w = tf.Variable(wb, name=f'{name}_w')
        # バイアス
        bb = tf.zeros([nf])
        self.bb = tf.Variable(bb, name=F'{name}_b')

    def call(self, x):
        w = self.w
        b = self.b
        # パーセプトロンを実行
        c = tf.matmul(x,w)+b
        return c
    
    def get_config(self, ):
        return super(FC, self).get_config()
    
# 1D畳み込み層(カーネルサイズ=1)
class Conv1d(tf.keras.layers.Layer):
    def __init__(self, n_inputs, n_feats, name, w_init_stdev=0.02):
        super(Conv1d, self).__init__(name=name)
        nx = n_inputs
        nf = n_feats
        # 重み
        wb = tf.random.normal([1, nx, nf], stddev=w_init_stdev)
        self.w = tf.Variable(wb, name=f'{name}_w')
        # バイアス
        bb = tf.random.normal([nf])
        self.b = tf.Variable(bb, name=f'{name}_b')

    def call(self, x):
        w = self.w
        b = self.b
        shape = x.shape.as_list()
        bs = -1 if shape[0] is None else shape[0] # バッチサイズ
        nx = shape[-1]
        nf = b.shape.as_list()[0]
        # カーネルサイズの幅に変形
        x = tf.reshape(x, [-1, nx])
        w = tf.reshape(w, [-1, nf])
        # 各単語の位置に対して演算
        c = tf.matmul(x,w)+b
        c = tf.reshape(c, (bs, shape[1], nf))
        return c
    
    def get_config(self, ):
        return super(Conv1d, self).get_config()
    
#Normalization層
class Normalization(tf.keras.layers.Layer):
    def __init__(self, n_inputs, name, axis=-1, epsilon=1e-5):
        super(Normalization, self).__init__(name=name)
        nx = n_inputs
        self.axis = axis
        self.epsilon = epsilon
        # 学習パラメータ
        gb = tf.ones([nx])
        self.g = tf.Variable(gb, name=f'{name}_g')
        bb = tf.zeros([nx])
        self.b = tf.Variable(bb, name=f'{name}_b')
    
    def call(self, x):
        g = self.g
        b = self.b
        # 平均値から二乗誤差
        u = tf.reduce_mean(x, axis=self.axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=self.axis, keepdims=True)
        # 平均値を二乗誤差の平方根で割る
        x = (x - u) * tf.math.rsqrt(s + self.epsilon)
        x = x*g + b
        return x
    
    def get_config(self, ):
        return super(Normalization, self).get_config()

