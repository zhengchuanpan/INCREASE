import time
import utils
import model
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime

# parameters
parser = argparse.ArgumentParser()
parser.add_argument('--h', default = 6, type = int)
parser.add_argument('--K', default = 15, type = int)
parser.add_argument('--T', default = 288, type = int)
parser.add_argument('--d', default = 64, type = int)
parser.add_argument('--lr', default = 0.001, type = float)
parser.add_argument('--epochs', default = 1000, type = int)
parser.add_argument('--patience', default = 10, type = int)
parser.add_argument('--data_file', default = '../data/metr-la.h5', type = str)
parser.add_argument('--dist_file', default = '../data/distance.csv', type = str)
##parser.add_argument('--test_file', default = '../data/test_node.npy', type = str)
parser.add_argument('--model_file', default = '../data/INCREASE15', type = str)
parser.add_argument('--log_file', default = '../data/log_train', type = str)
args = parser.parse_args()
log = open(args.log_file, 'w')

# load data
start = time.time()
utils.log_string(log, 'loading data...')
(train_x_gp_fw, train_x_gp_bw, train_gp_fw, train_gp_bw, train_TE, train_y,
 val_x_gp_fw, val_x_gp_bw, val_gp_fw, val_gp_bw, val_TE, val_y,
 test_x_gp_fw, test_x_gp_bw, test_gp_fw, test_gp_bw, test_TE, test_y) \
 = utils.load_data(args)
mean, std = np.mean(train_y), np.std(train_y)
utils.log_string(log, 'train_x_gp_fw: %s\ttrain_x_gp_bw: %s\ttrain_y: %s' %
                 (train_x_gp_fw.shape, train_x_gp_bw.shape, train_y.shape))
utils.log_string(log, 'val_x_gp_fw:   %s\tval_x_gp_bw:   %s\tval_y:   %s' %
                 (val_x_gp_fw.shape, val_x_gp_bw.shape, val_y.shape))
utils.log_string(log, 'test_x_gp_fw:  %s\ttest_x_gp_bw:  %s\ttest_y:  %s' %
                 (test_x_gp_fw.shape, test_x_gp_bw.shape, test_y.shape))
utils.log_string(log, 'mean: %.2f, std: %.2f' % (mean, std))
utils.log_string(log, 'data loaded!')

# train model
utils.log_string(log, 'compling model...')
x_gp_fw, x_gp_bw, gp_fw, gp_bw, TE, label = model.placeholder(args.h)
pred = model.model(x_gp_fw, x_gp_bw, gp_fw, gp_bw, TE, args.T, args.d, mean, std)
loss = model.mse_loss(pred, label)
tf.compat.v1.add_to_collection('pred', pred)
tf.compat.v1.add_to_collection('loss', loss)
optimizer = tf.compat.v1.train.AdamOptimizer(args.lr)
global_step = tf.Variable(0, trainable = False)
train_op = optimizer.minimize(loss, global_step = global_step)
parameters = 0
for variable in tf.compat.v1.trainable_variables():
    parameters += np.product([x.value for x in variable.get_shape()])
utils.log_string(log, 'total trainable parameters: {:,}'.format(parameters))
utils.log_string(log, 'model compiled!')
saver = tf.compat.v1.train.Saver()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)
sess.run(tf.compat.v1.global_variables_initializer())
utils.log_string(log, '**** training model ****')
train_time, val_time = [], []
val_loss_min = np.inf
wait = 0
for epoch in range(args.epochs):
    if wait >= args.patience:
        print('early stop at epoch: %d' % (epoch))
        break
    # train loss
    train_loss = []
    start_idx = np.random.choice(range(args.h))
    num_train = (train_x_gp_fw.shape[1] - start_idx) // args.h
    end_idx = start_idx + num_train * args.h
    train_x_gp_fw_epoch = train_x_gp_fw[:, start_idx : end_idx]
    train_x_gp_bw_epoch = train_x_gp_bw[:, start_idx : end_idx]
    train_x_gp_fw_epoch = np.reshape(
        train_x_gp_fw_epoch, newshape = (-1, num_train, args.h, args.K, 1))
    train_x_gp_bw_epoch = np.reshape(
        train_x_gp_bw_epoch, newshape = (-1, num_train, args.h, args.K, 1))
    train_TE_epoch = train_TE[:, start_idx : end_idx]
    train_TE_epoch = np.reshape(
        train_TE_epoch, newshape = (1, num_train, args.h))
    train_y_epoch = train_y[:, start_idx : end_idx]
    train_y_epoch = np.reshape(
        train_y_epoch, newshape = (-1, num_train, args.h, 1))
    permutation = np.random.permutation(num_train)
    train_x_gp_fw_epoch = train_x_gp_fw_epoch[:, permutation]
    train_x_gp_bw_epoch = train_x_gp_bw_epoch[:, permutation]
    train_TE_epocph = train_TE_epoch[:, permutation]
    train_y_epoch = train_y_epoch[:, permutation]
    t1 = time.time()
    for i in range(num_train):
        feed_dict = {
            x_gp_fw: train_x_gp_fw_epoch[:, i],
            x_gp_bw: train_x_gp_bw_epoch[:, i],
            TE: train_TE_epoch[:, i],
            gp_fw: train_gp_fw,
            gp_bw: train_gp_bw,
            label: train_y_epoch[:, i]}
        _, loss_batch = sess.run([train_op, loss], feed_dict = feed_dict)
        train_loss.append(loss_batch)
    t2 = time.time()
    train_time.append(t2 - t1)
    train_loss = np.mean(train_loss)     
    # val loss
    val_loss = []
    num_val = test_x_gp_fw.shape[1] // args.h
    t1 = time.time()
    for i in range(num_val):
        feed_dict = {
            x_gp_fw: test_x_gp_fw[:, i * args.h : (i + 1) * args.h],
            x_gp_bw: test_x_gp_bw[:, i * args.h : (i + 1) * args.h],
            TE: test_TE[:, i * args.h : (i + 1) * args.h],
            gp_fw: test_gp_fw,
            gp_bw: test_gp_bw,
            label: test_y[:, i * args.h : (i + 1) * args.h]}
        loss_batch = sess.run(loss, feed_dict = feed_dict)
        val_loss.append(loss_batch)
    t2 = time.time()
    val_time.append(t2 - t1)
    val_loss = np.mean(val_loss)
    utils.log_string(
        log, '{} | epoch: {:03d}/{}, '.format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            epoch + 1, args.epochs) + \
        'train_time: {:.2f}s, train_loss: {:.3f}, '.format(
            train_time[-1], train_loss) + \
        'val_time: {:.2f}s, val_loss: {:.3f}'.format(
            val_time[-1], val_loss))
    if val_loss <= val_loss_min:
        utils.log_string(
            log, 'val loss decrease from %.3f to %.3f, saving model to %s' %
            (val_loss_min, val_loss, args.model_file))
        val_loss_min = val_loss
        saver.save(sess, args.model_file)
        wait = 0
    else:
        wait += 1
utils.log_string(
    log, 'training finished, average train time: {:.2f}s, '.format(
        np.mean(train_time)) + 'average val time: {:.2f}s, '.format(
            np.mean(val_time)) + 'min val loss: {:.2f}'.format(val_loss_min))

# test model
utils.log_string(log, '**** testing model ****')
utils.log_string(log, 'loading model from %s' % args.model_file)
saver = tf.compat.v1.train.import_meta_graph(args.model_file + '.meta')
saver.restore(sess, args.model_file)
utils.log_string(log, 'model restored!')
utils.log_string(log, 'evaluating...')
test_pred = []
num_test = test_x_gp_fw.shape[1] // args.h
for i in range(num_test):
    feed_dict = {
        x_gp_fw: test_x_gp_fw[:, i * args.h : (i + 1) * args.h],
        x_gp_bw: test_x_gp_bw[:, i * args.h : (i + 1) * args.h],
        TE: test_TE[:, i * args.h : (i + 1) * args.h],
        gp_fw: test_gp_fw,
        gp_bw: test_gp_bw}
    pred_batch = sess.run(pred, feed_dict = feed_dict)
    test_pred.append(pred_batch)
test_pred = np.concatenate(test_pred, axis = 1)
num_res = test_y.shape[1] - test_pred.shape[1]
if num_res > 0:
    pred_batch = sess.run(
        pred, feed_dict = {
            x_gp_fw: test_x_gp_fw[:, -args.h :],
            x_gp_bw: test_x_gp_bw[:, -args.h :],
            TE: test_TE[:, -args.h :],
            gp_fw: test_gp_fw,
            gp_bw: test_gp_bw})
    pred_batch = pred_batch[:, -num_res :]
    test_pred = np.concatenate((test_pred, pred_batch), axis = 1)
# metric
test_rmse, test_mae, test_mape, test_r2 = utils.metric(test_pred, test_y)
utils.log_string(
    log, 'test_rmse: %.3f, test_mae: %.3f, test_mape: %.3f, test_r2: %.3f' %
    (test_rmse, test_mae, test_mape, test_r2))
end = time.time()
utils.log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
log.close()
sess.close()
