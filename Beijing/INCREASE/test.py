import time
import utils
import argparse
import numpy as np
import tensorflow as tf

# parameters
parser = argparse.ArgumentParser()
parser.add_argument('--h', default = 24, type = int)
parser.add_argument('--K', default = 5, type = int)
parser.add_argument('--T', default = 24, type = int)
parser.add_argument('--data_file', default = '../data/Beijing.h5', type = str)
parser.add_argument('--POI_file', default = '../data/POI.npy', type = str)
parser.add_argument('--dist_file', default = '../data/distance.npy', type = str)
parser.add_argument('--test_file', default = '../data/test_node.npy', type = str)
parser.add_argument('--model_file', default = '../data/INCREASE_SP+FS_pretrained', type = str)
parser.add_argument('--log_file', default = '../data/log_test(SP+FS)', type = str)
args = parser.parse_args()
log = open(args.log_file, 'w')

# load data
start = time.time()
utils.log_string(log, 'loading data...')
(train_x_gp, train_gp, train_x_fs, train_fs, train_TE, train_y,
 val_x_gp, val_gp, val_x_fs, val_fs, val_TE, val_y,
 test_x_gp, test_gp, test_x_fs, test_fs, test_TE, test_y) \
 = utils.load_data(args)
utils.log_string(log, 'train_x: %s\ttrain_y: %s' %
                 (train_x_gp.shape, train_y.shape))
utils.log_string(log, 'val_x:   %s\tval_y:   %s' %
                 (val_x_gp.shape, val_y.shape))
utils.log_string(log, 'test_x:  %s\ttest_y:  %s' %
                 (test_x_gp.shape, test_y.shape))
utils.log_string(log, 'data loaded!')

# test model
utils.log_string(log, '**** testing model ****')
utils.log_string(log, 'loading model from %s' % args.model_file)
graph = tf.Graph()
with graph.as_default():
    saver = tf.compat.v1.train.import_meta_graph(args.model_file + '.meta')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
with tf.compat.v1.Session(graph = graph, config = config) as sess:
    saver.restore(sess, args.model_file)
    parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        parameters += np.product([x.value for x in variable.get_shape()])
    utils.log_string(log, 'trainable parameters: {:,}'.format(parameters))
    pred = graph.get_collection(name = 'pred')[0]
    utils.log_string(log, 'model restored!')
    utils.log_string(log, 'evaluating...')
    # test set
    test_pred = []
    num_test = test_x_gp.shape[1] // args.h
    for i in range(num_test):
        feed_dict = {
            'x_gp:0': test_x_gp[:, i * args.h : (i + 1) * args.h],
            'x_fs:0': test_x_fs[:, i * args.h : (i + 1) * args.h],
            'TE:0': test_TE[:, i * args.h : (i + 1) * args.h],
            'gp:0': test_gp,
            'fs:0': test_fs}
        pred_batch = sess.run(pred, feed_dict = feed_dict)
        test_pred.append(pred_batch)
    test_pred = np.concatenate(test_pred, axis = 1)
    num_res = test_y.shape[1] - test_pred.shape[1]
    if num_res > 0:
        pred_batch = sess.run(
            pred, feed_dict = {
                'x_gp:0': test_x_gp[:, -args.h :],
                'x_fs:0': test_x_fs[:, -args.h :],
                'TE:0': test_TE[:, -args.h :],
                'gp:0': test_gp,
                'fs:0': test_fs})
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
