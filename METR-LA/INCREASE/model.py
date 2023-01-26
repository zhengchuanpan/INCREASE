import tensorflow as tf

def placeholder(h):
    '''
    x:      [N_target, h, K, 1]
    weight: [N_target, 1, 1, K]
    TE:     [1, h]
    label:  [N_target, h, 1]
    '''
    x_gp_fw = tf.compat.v1.placeholder(
        shape = (None, h, None, 1), dtype = tf.float32, name = 'x_gp_fw')
    x_gp_bw = tf.compat.v1.placeholder(
        shape = (None, h, None, 1), dtype = tf.float32, name = 'x_gp_bw')
    gp_fw = tf.compat.v1.placeholder(
        shape = (None, 1, 1, None), dtype = tf.float32, name = 'gp_fw')
    gp_bw = tf.compat.v1.placeholder(
        shape = (None, 1, 1, None), dtype = tf.float32, name = 'gp_bw')
    TE = tf.compat.v1.placeholder(
        shape = (1, h), dtype = tf.int32, name = 'TE')
    label = tf.compat.v1.placeholder(
        shape = (None, h, 1), dtype = tf.float32, name = 'label')
    return x_gp_fw, x_gp_bw, gp_fw, gp_bw, TE, label

def FC(x, units, activations, use_bias = True):
    for unit, activation in zip(units, activations):
        x = tf.keras.layers.Dense(
            units = unit, activation = activation, use_bias = use_bias)(x)
    return x

def model(x_gp_fw, x_gp_bw, gp_fw, gp_bw, TE, T, d, mean, std):
    N_target = tf.shape(x_gp_fw)[0]
    h = x_gp_fw.get_shape()[1]
    K = gp_fw.get_shape()[-1]
    # input
    x_gp_fw = (x_gp_fw - mean) / std
    x_gp_bw = (x_gp_bw - mean) / std
    x_gp_fw = FC(x_gp_fw, units = [d, d], activations = ['relu', None])
    x_gp_bw = FC(x_gp_bw, units = [d, d], activations = ['relu', None])
    # spatial aggregation
    gp_fw = tf.tile(gp_fw, multiples = (1, h, 1, 1))
    gp_bw = tf.tile(gp_bw, multiples = (1, h, 1, 1))
    y_gp_fw = tf.matmul(gp_fw, x_gp_fw)
    y_gp_bw = tf.matmul(gp_bw, x_gp_bw)
    y_gp_fw = FC(y_gp_fw, units = [d, d], activations = ['relu', None])
    y_gp_bw = FC(y_gp_bw, units = [d, d], activations = ['relu', None])
    x_gp_fw = FC(x_gp_fw, units = [d, d], activations = ['relu', None])
    x_gp_bw = FC(x_gp_bw, units = [d, d], activations = ['relu', None])
    x_gp_fw = tf.abs(y_gp_fw - x_gp_fw)
    x_gp_bw = tf.abs(y_gp_bw - x_gp_bw)
    x_gp_fw = tf.matmul(gp_fw, x_gp_fw)
    x_gp_bw = tf.matmul(gp_bw, x_gp_bw)
    x_gp_fw = FC(x_gp_fw, units = [d, d], activations = ['relu', 'tanh'])
    x_gp_bw = FC(x_gp_bw, units = [d, d], activations = ['relu', 'tanh'])
    y_gp_fw = FC(y_gp_fw, units = [d], activations = [None])
    y_gp_bw = FC(y_gp_bw, units = [d], activations = [None])
    y_gp_fw = x_gp_fw + y_gp_fw
    y_gp_bw = x_gp_bw + y_gp_bw
    y_gp_fw = FC(y_gp_fw, units = [d, d], activations = ['relu', None])
    y_gp_bw = FC(y_gp_bw, units = [d, d], activations = ['relu', None])    
    # temporal modeling
    TE = tf.one_hot(TE, depth = T)
    TE = FC(TE, units = [d, d], activations = ['relu', None])
    TE = tf.tile(TE, multiples = (N_target, 1, 1))
    y = tf.concat((y_gp_fw, y_gp_bw), axis = -1)
    y = tf.squeeze(y, axis = 2)
    y = FC(y, units = [d, d], activations = ['relu', None])
    x = tf.concat((x_gp_fw, x_gp_bw), axis = -1)
    x = tf.squeeze(x, axis = 2)
    x = FC(x, units = [d, d], activations = ['relu', None])
    g1 = FC(x, units = [d, d], activations = ['relu', 'relu'])
    g1 = 1 / tf.exp(g1)
    y = g1 * y
    y = tf.concat((y, TE), axis = -1)
    pred = []
    cell = tf.nn.rnn_cell.GRUCell(num_units = d)
    state = tf.zeros(shape = (N_target, d))
    for i in range(h):
        if i == 0:
            g2 = tf.layers.dense(state, units = d, name = 'g2')
            g2 = tf.nn.relu(g2)
            g2 = 1 / tf.exp(g2)
            state = g2 * state 
            state, _ = cell.__call__(y[:, i], state)
            pred.append(tf.expand_dims(state, axis = 1))
        else:
            g2 = tf.layers.dense(x[:, i - 1], units = d, name = 'g2', reuse = True)
            g2 = tf.nn.relu(g2)
            g2 = 1 / tf.exp(g2)
            state = g2 * state 
            state, _ = cell.__call__(y[:, i], state)
            pred.append(tf.expand_dims(state, axis = 1))
    pred = tf.concat(pred, axis = 1)
    # output
    pred = FC(pred, units = [d, d, 1], activations = ['relu', 'relu', None])
    return pred * std + mean
    
def mse_loss(pred, label):
    mask = tf.not_equal(label, 0)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.compat.v2.where(tf.math.is_nan(mask), x = 0., y = mask)
    loss = tf.subtract(pred, label) ** 2
    loss *= mask
    loss = tf.compat.v2.where(tf.math.is_nan(loss), x = 0., y = loss)
    loss = tf.reduce_mean(loss)
    return loss    
    
