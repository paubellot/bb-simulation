import os
import numpy as np
import Utils
from models import Train

fdir = '/home/data/biobank/'

def data_rand(k=50000):
    snp_list = np.random.permutation(500000)[0:k]
    x_train, _, _ = Utils.readgbinfile(fdir + 'genosTRN.bin', snp_list=snp_list, std=False)
    x_test, _, _ = Utils.readgbinfile(fdir + 'genosTST.bin', snp_list=snp_list, std=False)
    return x_train, x_test, snp_list


def syn_data_lin(x_train, x_test, shape, scale, h=0.45):
    n_p, n_f =  x_train.shape

    s = np.random.gamma(shape, scale, n_f)
    w = s * np.random.permutation(np.asanyarray([-1, 1]*(n_f/2)))
    w = w/sum(w)
    s = np.sqrt((1-h)/h*np.dot(x_train,w).std()**2)
    y_tr = np.dot(x_train, w) + s*np.random.randn(n_p)
    s = np.sqrt((1-h)/h*np.dot(x_test,w).std()**2)
    y_tst = np.dot(x_test, w)+s*np.random.randn(x_test.shape[0])
    return y_tr, y_tst, w


def syn_data_epis(x_train, x_test,shape,scale,h=0.45):
    n_p, n_f = x_train.shape
    ind = np.random.permutation(n_f)
    ind = ind.reshape(n_f / 2, 2)
    y = np.zeros((x_train.shape[0], 1)).ravel()
    t = np.zeros((x_test.shape[0], 1)).ravel()
    for i in range(0, ind.shape[0]):
        aux = np.prod(x_train[:, ind[i, :]], axis=1)
        E = np.random.gamma(shape, scale, 1) * np.random.permutation(np.asanyarray([-1, 1]))[0]
        y[np.where(aux != 0)[0]] += E
        aux = np.prod(x_test[:, ind[i, :]], axis=1)
        t[np.where(aux != 0)[0]] += E
    p = np.sqrt(np.var(y) * (1. - h) / h)
    ytr = y + p * np.random.randn(y.shape[0], 1).ravel()
    h2 = np.var(y) / np.var(ytr)
    print h2
    ytst = t + p * np.random.randn(t.shape[0], 1).ravel()
    return ytr, ytst


def computeCor(y_tr):
    from scipy.stats import pearsonr
    p = 589028
    l = range(0,p)
    n = p/100
    chunks = [l[i:i + n] for i in range(0, len(l), n)]
    result_all = []
    for c in range(0,len(chunks)):
        print("reading chunk %d" % c)
        (X_train, n_train, p) = Utils.readgbinfile(fdir + 'genosTRN.bin',std=False,snp_list=chunks[c])
        for i in range(0, X_train.shape[1]):
            result_all.append(pearsonr(X_train[:, i], y_tr))
    return  np.asarray(result_all)


def readGWAS(C,k):
    C[C != C] = 0
    snp_list = abs(C[:, 0]).ravel().argsort()[::-1]
    snp_list = snp_list[0:k]
    x_train, _, _ = Utils.readgbinfile(fdir + 'genosTRN.bin', snp_list=snp_list, std=False)
    x_test, _, _ = Utils.readgbinfile(fdir + 'genosTST.bin', snp_list=snp_list, std=False)
    return x_train, x_test


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="""Simulated data """)
    parser.add_argument('--method',
                        type=str,
                        default="mlp1",
                        help='Method: mlp1, mlp2, lasso or ridge')

    parser.add_argument('--k',
                        type=int,
                        default=10000,
                        help='number of snps')

    parser.add_argument('--loci',
                        type=int,
                        default=100,
                        help='number of causal loci snps')

    parser.add_argument('--model',
                        type=str,
                        default="linear",
                        help='Phenotype model: linear or epistasia. Default linear')

    parser.add_argument('--recreation',
                        type=bool,
                        default=False,
                        help='force recreation (bool): default False')

    parser.add_argument('--shape',
                        type=float,
                        default=5,
                        help='np.random.gamma shape: 5')

    parser.add_argument('--scale',
                        type=float,
                        default=0.25,
                        help='np.random.gamma scale: 0.25')

    parser.add_argument('--h2',
                        type=float,
                        default=0.45,
                        help='hearability: 0.45')

    args = parser.parse_args()
    if args.model =="linear":
        ytr_name = "data/y_train_linear_"+str(args.loci)+".txt"
        ytst_name = "data/y_test_linear_"+str(args.loci)+".txt"
    else:
        ytr_name = "data/y_train_epis_" + str(args.loci) + ".txt"
        ytst_name = "data/y_test_epis_" + str(args.loci) + ".txt"

    C_name = "data/Corr_"+str(args.loci)+".csv"

    if os.path.exists(ytr_name) and not args.recreation:
        print("Generating phenotype")
        x_train, x_test, snp_list = data_rand(k=args.loci)
        if args.model == "linear":
            print("Linear model")
            y_tr, y_tst, w = syn_data_lin(x_train, x_test, shape=args.shape, scale=args.scale)
        else:
            print("Epistasia model")
            y_tr, y_tst = syn_data_epis(x_train, x_test, args.loci, shape=args.shape, scale=args.scale)
        np.savetxt(ytr_name, y_tr, delimiter=',')
        np.savetxt(ytst_name, y_tst, delimiter=',')
    if os.path.exists(C_name) and not args.recreation:
        C = computeCor(y_tr)
        np.savetxt(C_name, C, delimiter=",")
    else:
        C = np.loadtxt(C_name, delimiter=",")

    xTr, xTst = readGWAS(C, args.k)
    r = Train(xTr, xTst, y_tr, y_tst, args.method)

    print("R2 tst: %5.3f." % r)
