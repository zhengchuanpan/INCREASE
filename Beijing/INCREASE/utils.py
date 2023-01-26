import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def metric(pred, truth):
    idx = truth > 0
    RMSE = np.sqrt(np.mean((pred[idx] - truth[idx]) ** 2))
    MAE = np.mean(np.abs(pred[idx] - truth[idx]))
    R2 = 1 - np.mean((pred[idx] - truth[idx]) ** 2) / \
         np.mean((truth[idx] - np.mean(truth[idx])) ** 2)
    # filter small values to make the MAPE more stable, same for all methods
    # constitute 90% of the valid values
    idx = truth > 10  
    MAPE = np.mean(np.abs(pred[idx] - truth[idx]) / truth[idx])
    return RMSE, MAE, MAPE, R2

def heterogeneous_relations(A, train_node, K):
    relation = np.zeros_like(A) - 1
    relation[:, train_node] = A[:, train_node] # only use the train nodes
    np.fill_diagonal(relation, val = -1) # delete self-loop connections
    Neighbor = np.argsort(-relation, axis = 1) # descending order
    Neighbor = Neighbor[:, : K]
    relation = -np.sort(-relation, axis = -1)
    relation = relation[:, : K]
    relation[relation < 0] = 0
    relation = relation / (1e-10 + np.sum(relation, axis = 1, keepdims = True))
    return relation, Neighbor

def load_data(args):
    # x
    df = pd.read_hdf(args.data_file)
    x = df.values.astype(np.float32).T
    N, num_interval = x.shape
    # TE
    TE = df.index
    TE = (TE.hour * 3600 + TE.minute * 60 + TE.second) // (24 * 3600 / args.T)
    TE = np.array(TE).astype(np.int32)
    TE = TE[np.newaxis]
    # node
    test_node = np.load(args.test_file)
    all_node = np.arange(N)
    train_node = np.setdiff1d(all_node, test_node)
    np.random.shuffle(train_node)
    # geo-proximity
    dist_mx = np.load(args.dist_file)
    std = np.std(dist_mx[train_node, :][:, train_node])
    A_gp = np.exp(-dist_mx ** 2 / std ** 2)
    gp, Neighbor_gp = heterogeneous_relations(A_gp, train_node, args.K)
    # function-similarity
    POI = np.load(args.POI_file).astype(np.float32)
    POI_train = POI[train_node]
    mean, std = np.mean(POI_train, axis = 0), np.std(POI_train, axis = 0)
    POI = (POI - mean) / std
    A_fs = np.zeros(shape = (N, N))
    for i in range(N):
        for j in range(N):
            A_fs[i, j] = pearsonr(POI[i], POI[j])[0]
    fs, Neighbor_fs = heterogeneous_relations(A_fs, train_node, args.K)
    # train/val/test
    num_train = int(0.7 * num_interval)
    num_val = int(0.2 * num_train)
    num_train -= num_val
    num_test = num_interval - num_train - num_val
    train_TE = TE[:, : num_train]
    val_TE = TE[:, num_train : num_train + num_val]
    test_TE = TE[:, -num_test :]
    train_x_gp = np.transpose(
        x[Neighbor_gp[train_node], : num_train, np.newaxis],
        axes = (0, 2, 1, 3))
    train_x_fs = np.transpose(
        x[Neighbor_fs[train_node], : num_train, np.newaxis],
        axes = (0, 2, 1, 3))
    train_y = x[train_node, : num_train, np.newaxis]
    val_x_gp = np.transpose(
        x[Neighbor_gp[train_node], num_train : num_train + num_val, np.newaxis],
        axes = (0, 2, 1, 3))
    val_x_fs = np.transpose(
        x[Neighbor_fs[train_node], num_train : num_train + num_val, np.newaxis],
        axes = (0, 2, 1, 3))
    val_y = x[train_node, num_train : num_train + num_val, np.newaxis]
    test_x_gp = np.transpose(
        x[Neighbor_gp[test_node], -num_test :, np.newaxis],
        axes = (0, 2, 1, 3))
    test_x_fs = np.transpose(
        x[Neighbor_fs[test_node], -num_test :, np.newaxis],
        axes = (0, 2, 1, 3))
    test_y = x[test_node, -num_test :, np.newaxis]
    train_gp = gp[train_node, np.newaxis, np.newaxis]
    train_fs = fs[train_node, np.newaxis, np.newaxis]
    val_gp = gp[train_node, np.newaxis, np.newaxis]
    val_fs = fs[train_node, np.newaxis, np.newaxis]
    test_gp = gp[test_node, np.newaxis, np.newaxis]
    test_fs = fs[test_node, np.newaxis, np.newaxis]
    return (train_x_gp, train_gp, train_x_fs, train_fs, train_TE, train_y,
            val_x_gp, val_gp, val_x_fs, val_fs, val_TE, val_y,
            test_x_gp, test_gp, test_x_fs, test_fs, test_TE, test_y)

