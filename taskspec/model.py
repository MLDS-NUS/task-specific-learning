import tensorflow as tf

def ResNet(n_dim, nodes, tau, scale_min, scale, name='_ResNet'):
    tau = tf.constant(tau, dtype=tf.float64)
    scale_min = tf.constant(scale_min, dtype=tf.float64)
    scale = tf.constant(scale, dtype=tf.float64)
    assert scale.shape == (n_dim,)
    assert scale_min.shape == (n_dim,)
    
    x = tf.keras.Input(shape=(n_dim,), name='input'+name)    
    xscale = (x - scale_min) / scale    
    h = tf.keras.layers.Dense(nodes, activation='elu', name='hidden'+name)(xscale)    
    y = tau * scale * tf.keras.layers.Dense(n_dim, name='output'+name)(h) + x

    model = tf.keras.Model(inputs=x, outputs=y)
    model.scale = scale
    model.scale_min = scale_min
    model.tau = tau
    return tf.keras.Model(inputs=x, outputs=y)

def DenseNet(n_dim, n_dim_out, nodes, scale, name='_FFN'):
    scale = tf.constant(scale, dtype=tf.float64)
    x = tf.keras.Input(shape=(n_dim,), name='input'+name)
    h = tf.keras.layers.Dense(nodes, activation='elu', name='dense'+name)(x/scale)
    y = tf.keras.layers.Dense(n_dim_out, name='output'+name)(h)
    return tf.keras.Model(inputs=x, outputs=y)

def PolyModel(n_dim, n_dim_out, powers, scale, name='_Linear'):
    scale = tf.constant(scale, dtype=tf.float64)
    x = tf.keras.Input(shape=(n_dim,), name='input'+name)
    px = tf.reduce_prod(tf.pow(tf.expand_dims(x/scale, axis=1), powers), axis=-1)
    y = tf.keras.layers.Dense(n_dim_out, name='output'+name)(px)
    return tf.keras.Model(inputs=x, outputs=y)

def DenseNet_Energy(n_dim, nodes, scale, a=0.0001, name='_EnergyModel'):
    scale = tf.constant(scale, dtype=tf.float64)
    x = tf.keras.Input(shape=(n_dim,), name='input'+name)
    xscale = x/scale
    
    a = tf.constant(a, dtype=tf.float64)
    x2 = tf.reduce_sum(tf.square(xscale), axis=-1, keepdims=True)    
    lx = tf.keras.layers.Dense(1, name='lx'+name)(xscale)
    lx2 = tf.reduce_sum(tf.square(lx), axis=-1, keepdims=True)
    
    h = tf.keras.layers.Dense(nodes, activation='elu', name='dense'+name)(xscale)
    y = tf.keras.layers.Dense(1, name='output'+name)(h)
    
    output = a*x2 + lx2 + y
    return tf.keras.Model(inputs=x, outputs=output)

class Grad_model(tf.keras.Model):
    def __init__(self, V_model):
        super(Grad_model, self).__init__()
        self.V_model = V_model
    
    @tf.function
    def call(self, x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.V_model(x)
        dy_dx = tape.gradient(y, x)
        return -dy_dx
    
