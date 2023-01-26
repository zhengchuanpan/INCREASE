import tensorflow as tf

def placeholder(h):
    '''
    x:      [N_target, h, K, 1]
    weight: [N_target, 1, 1, K]
    TE:     [1, h]
    label:  [N_target, h, 1]
    '''
    x_gp = tf.compat.v1.placeholder(
        shape = (None, h, None, 1), dtype = tf.float32, name = 'x_gp')
    x_fs = tf.compat.v1.placeholder(
        shape = (None, h, None, 1), dtype = tf.float32, name = 'x_fs')
    gp = tf.compat.v1.placeholder(
        shape = (None, 1, 1, None), dtype = tf.float32, name = 'gp')
    fs = tf.compat.v1.placeholder(
        shape = (None, 1, 1, None), dtype = tf.float32, name = 'fs')
    TE = tf.compat.v1.placeholder(
        shape = (1, h), dtype = tf.int32, name = 'TE')
    label = tf.compat.v1.placeholder(
        shape = (None, h, 1), dtype = tf.float32, name = 'label')
    return x_gp, x_fs, gp, fs, TE, label

def FC(x, units, activations, use_bias = True):
    for unit, activation in zip(units, activations):
        x = tf.keras.layers.Dense(
            units = unit, activation = activation, use_bias = use_bias)(x)
    return x

def attention(x, d):
    query = FC(x, units = [d, d], activations = ['relu', None])
    key = FC(x, units = [d, d], activations = ['relu', None])
    value = FC(x, units = [d, d], activations = ['relu', None])
    att = tf.matmul(query, key, transpose_b = True)
    att = tf.nn.softmax(att, axis = -1)
    x = tf.matmul(att, value)
    return x

def attention(x):
    '''
    x: [None, P, R, d]
    '''
    d = x.get_shape()[-1].value
    a = FC(x, units = [d], activations = ['tanh'])
    a = FC(a, units = [1], activations = [None], use_bias = False)
    a = tf.nn.softmax(tf.transpose(a, perm = (0, 1, 3, 2)))
    x = tf.matmul(a, x)
    x = tf.squeeze(x, axis = 2)
    return x

def model(x_gp, x_fs, gp, fs, TE, T, d, mean, std):
    N_target = tf.shape(x_gp)[0]
    h = x_gp.get_shape()[1]
    K = gp.get_shape()[-1]
    # input
    x_gp = (x_gp - mean) / std
    x_fs = (x_fs - mean) / std
    x_gp = FC(x_gp, units = [d, d], activations = ['relu', None])
    x_fs = FC(x_fs, units = [d, d], activations = ['relu', None])
    # spatial aggregation
    gp = tf.tile(gp, multiples = (1, h, 1, 1))
    fs = tf.tile(fs, multiples = (1, h, 1, 1))    
    y_gp = tf.matmul(gp, x_gp)
    y_fs = tf.matmul(fs, x_fs)
    x_gp = FC(x_gp, units = [d, d], activations = ['relu', None])
    x_fs = FC(x_fs, units = [d, d], activations = ['relu', None])
    y_gp = FC(y_gp, units = [d, d], activations = ['relu', None])
    y_fs = FC(y_fs, units = [d, d], activations = ['relu', None])
    x_gp = tf.abs(y_gp - x_gp)
    x_fs = tf.abs(y_fs - x_fs)
    x_gp = tf.matmul(gp, x_gp)
    x_fs = tf.matmul(fs, x_fs)
    x_gp = FC(x_gp, units = [d, d], activations = ['relu', 'tanh'])
    x_fs = FC(x_fs, units = [d, d], activations = ['relu', 'tanh'])
    y_gp = FC(y_gp, units = [d], activations = [None])
    y_fs = FC(y_fs, units = [d], activations = [None])
    y_gp = x_gp + y_gp
    y_fs = x_fs + y_fs
    x_gp = FC(x_gp, units = [d, d], activations = ['relu', None])
    x_fs = FC(x_fs, units = [d, d], activations = ['relu', None])
    y_gp = FC(y_gp, units = [d, d], activations = ['relu', None])
    y_fs = FC(y_fs, units = [d, d], activations = ['relu', None])    
    # temporal modeling
    TE = tf.one_hot(TE, depth = T)
    TE = FC(TE, units = [d, d], activations = ['relu', None])
    TE = tf.tile(TE, multiples = (N_target, 1, 1))
    y_gp = tf.squeeze(y_gp, axis = 2)
    y_fs = tf.squeeze(y_fs, axis = 2)
    x_gp = tf.squeeze(x_gp, axis = 2)
    x_fs = tf.squeeze(x_fs, axis = 2)
    g1_gp = FC(x_gp, units = [d, d], activations = ['relu', 'relu'])
    g1_gp = 1 / tf.exp(g1_gp)
    y_gp = g1_gp * y_gp
    y_gp = tf.concat((y_gp, TE), axis = -1)
    g1_fs = FC(x_fs, units = [d, d], activations = ['relu', 'relu'])
    g1_fs = 1 / tf.exp(g1_fs)
    y_fs = g1_fs * y_fs
    y_fs = tf.concat((y_fs, TE), axis = -1)
    pred_gp, pred_fs = [], []
    cell_gp = tf.nn.rnn_cell.GRUCell(num_units = d)
    cell_fs = tf.nn.rnn_cell.GRUCell(num_units = d)
    state = tf.zeros(shape = (N_target, d))
    for i in range(h):
        if i == 0:
            g2_gp = tf.layers.dense(state, units = d, name = 'g2_gp')
            g2_gp = tf.nn.relu(g2_gp)
            g2_gp = 1 / tf.exp(g2_gp)
            state_gp = g2_gp * state 
            state_gp, _ = cell_gp.__call__(y_gp[:, i], state_gp)
            pred_gp.append(tf.expand_dims(state_gp, axis = 1))
            g2_fs = tf.layers.dense(state, units = d, name = 'g2_fs')
            g2_fs = tf.nn.relu(g2_fs)
            g2_fs = 1 / tf.exp(g2_fs)
            state_fs = g2_fs * state 
            state_fs, _ = cell_fs.__call__(y_fs[:, i], state_fs)
            pred_fs.append(tf.expand_dims(state_fs, axis = 1))
        else:
            g2_gp = tf.layers.dense(x_gp[:, i - 1], units = d, name = 'g2_gp', reuse = True)
            g2_gp = tf.nn.relu(g2_gp)
            g2_gp = 1 / tf.exp(g2_gp)
            state_gp = g2_gp * state_gp 
            state_gp, _ = cell_gp.__call__(y_gp[:, i], state_gp)
            pred_gp.append(tf.expand_dims(state_gp, axis = 1))
            g2_fs = tf.layers.dense(x_fs[:, i - 1], units = d, name = 'g2_fs', reuse = True)
            g2_fs = tf.nn.relu(g2_fs)
            g2_fs = 1 / tf.exp(g2_fs)
            state_fs = g2_fs * state_fs 
            state_fs, _ = cell_fs.__call__(y_fs[:, i], state_fs)
            pred_fs.append(tf.expand_dims(state_fs, axis = 1))
    pred_gp = tf.concat(pred_gp, axis = 1)
    pred_fs = tf.concat(pred_fs, axis = 1)
    # multi-relation fusion
    pred_gp = tf.expand_dims(pred_gp, axis = 2)
    pred_fs = tf.expand_dims(pred_fs, axis = 2)
    pred = tf.concat((pred_gp, pred_fs), axis = 2)
    pred = attention(pred)
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
    
