import numpy as np
import pandas as pd
rand = np.random.RandomState(0)

def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def metric(pred, truth):
    idx = truth > 0
    RMSE = np.sqrt(np.mean((pred[idx] - truth[idx]) ** 2))
    MAE = np.mean(np.abs(pred[idx] - truth[idx]))
    MAPE = np.mean(np.abs(pred[idx] - truth[idx]) / truth[idx])
    R2 = 1 - np.mean((pred[idx] - truth[idx]) ** 2) / \
         np.mean((truth - np.mean(truth)) ** 2)
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
    nodes = list(df.columns)
    x = df.values.astype(np.float32).T
    N, num_interval = x.shape
    # TE
    TE = df.index
    TE = (TE.hour * 3600 + TE.minute * 60 + TE.second) // (24 * 3600 / args.T)
    TE = np.array(TE).astype(np.int32)
    TE = TE[np.newaxis]
    # node
    test_node = rand.choice(list(range(0, N)),int(0.5 * N),replace=False)
    print(test_node)
##    test_node = np.load(args.test_file)
    train_node = np.setdiff1d(np.arange(N), test_node)
    np.random.shuffle(train_node)
    # Geo-proximity
    dist = np.loadtxt(args.dist_file, delimiter = ',', skiprows = 1)
    dist_mx = np.zeros(shape = (N, N), dtype = np.float32)
    dist_mx[:] = np.inf
    for row in dist:
        node1 = str(int(row[0]))
        node2 = str(int(row[1]))
        if node1 in nodes and node2 in nodes:
            index1 = nodes.index(node1)
            index2 = nodes.index(node2)
            dist_mx[index1, index2] = row[2]
    dist_train = dist_mx[train_node, :][:, train_node]
    std = np.std(dist_train[~np.isinf(dist_train)])
    A_gp = np.exp(-dist_mx ** 2 / std ** 2)
    gp_fw, Neighbor_gp_fw = heterogeneous_relations(A_gp, train_node, args.K)  
    gp_bw, Neighbor_gp_bw = heterogeneous_relations(A_gp.T, train_node, args.K)   
    # train/val/test
    num_train = int(0.5 * num_interval)
    num_val = int(0.2 * num_train)
##    num_train -= num_val
    num_test = num_interval - num_train - num_val
    train_TE = TE[:, : num_train]
    val_TE = TE[:, num_train : num_train + num_val]
    test_TE = TE[:, -num_test :]
    train_x_gp_fw = np.transpose(
        x[Neighbor_gp_fw[train_node], : num_train, np.newaxis],
        axes = (0, 2, 1, 3))
    train_x_gp_bw = np.transpose(
        x[Neighbor_gp_bw[train_node], : num_train, np.newaxis],
        axes = (0, 2, 1, 3))
    train_y = x[train_node, : num_train, np.newaxis]
    val_x_gp_fw = np.transpose(
        x[Neighbor_gp_fw[train_node], num_train : num_train + num_val, np.newaxis],
        axes = (0, 2, 1, 3))
    val_x_gp_bw = np.transpose(
        x[Neighbor_gp_bw[train_node], num_train : num_train + num_val, np.newaxis],
        axes = (0, 2, 1, 3))
    val_y = x[train_node, num_train : num_train + num_val, np.newaxis]
    test_x_gp_fw = np.transpose(
        x[Neighbor_gp_fw[test_node], -num_test :, np.newaxis],
        axes = (0, 2, 1, 3))
    test_x_gp_bw = np.transpose(
        x[Neighbor_gp_bw[test_node], -num_test :, np.newaxis],
        axes = (0, 2, 1, 3))
    test_y = x[test_node, -num_test :, np.newaxis]
    train_gp_fw = gp_fw[train_node, np.newaxis, np.newaxis]
    train_gp_bw = gp_bw[train_node, np.newaxis, np.newaxis]
    val_gp_fw = gp_fw[train_node, np.newaxis, np.newaxis]
    val_gp_bw = gp_bw[train_node, np.newaxis, np.newaxis]
    test_gp_fw = gp_fw[test_node, np.newaxis, np.newaxis]
    test_gp_bw = gp_bw[test_node, np.newaxis, np.newaxis]
    return (train_x_gp_fw, train_x_gp_bw, train_gp_fw, train_gp_bw, train_TE, train_y,
            val_x_gp_fw, val_x_gp_bw, val_gp_fw, val_gp_bw, val_TE, val_y,
            test_x_gp_fw, test_x_gp_bw, test_gp_fw, test_gp_bw, test_TE, test_y)

