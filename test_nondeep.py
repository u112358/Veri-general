
import ReadBatches as data_reader
import numpy as np
from six.moves import xrange
from sklearn.decomposition.pca import PCA

batch_size = 20
address="../../dataset/SUNReady/"
learning_rate = 0.0001

max_iter = 1

alpha = 7e3


reader = data_reader.BatchDatset(address, batch_size)

B = reader.sim_pairs[:, 4096:4198]

pca = PCA(n_components=20)
pca.fit(B)

W2 = pca.transform(np.eye(np.shape(B)[1]))
W2 = np.mat(W2)
for i in xrange(max_iter):
    A = np.mat(reader.sim_pairs[:, :4096])
    B = np.mat(reader.sim_pairs[:, 4096:4198])
    # C = np.mat(reader.dsim_pairs[:, :4096])
    D = np.mat(reader.sim_pairs[:, 4198:4300])
    # fix w2 updata w1
    right = (A.T * B - A.T * D) * W2
    # left = (A.T * A - A.T * A + 100*np.mat(np.eye(np.shape(C)[1])))
    W1 = right/alpha

    right = (B.T * A - D.T * A) * W1
    left = (B.T * B - D.T * D + alpha*np.mat(np.eye(np.shape(D)[1])))
    W2 = left.I * right
    # reader.sim_pairs = reader.train_batch_generation()
    print("-------------------------------------------------------------"+str(i))

test_data_A = reader.test_pairs[:, :4096]
test_data_B = reader.test_pairs[:, 4096:4096+102]
test_label = reader.test_pairs[:, 4096+102:]
count = 0
for i in xrange(200):
    A = np.mat(test_data_A[i*10:(i+1)*10,:])
    AW = A * W1
    B = np.mat(test_data_B[i*10:(i+1)*10,:])
    BW = B * W2
    K = test_label[i*10:(i+1)*10,:]
    loss = np.array(AW-BW)
    loss = np.sqrt(np.sum(np.square(loss), axis=1))
    pos = np.argmin(loss, axis=0)
    if K[pos]>0.5:
        count += 1
acc = count/200
print("acc="+ str(acc))



