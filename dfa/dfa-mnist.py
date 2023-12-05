import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from scipy.special import expit
from sklearn.metrics import log_loss

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def forward_pass(W1, W2, b1, b2, x):
    '''This is the forward pass. It is equal for any
    training algorithm. It's just one hidden layer
    with tanh activation function and sigmoid on the
    output layer'''
    # if the input is a batch, I have to tile as many
    # b1 and b2 as the batch size
    a1 = np.matmul(W1, x)+np.tile(b1, x.shape[1])
    h1 = np.tanh(a1)
    a2 = np.matmul(W2, h1)+np.tile(b2, x.shape[1])
    y_hat = expit(a2)
    return a1, h1, a2, y_hat

def backprop_backward_pass(e, h1, W2, a1, x):
    dW2 = -np.matmul(e, np.transpose(h1))
    da1 = np.matmul(np.transpose(W2), e)*(1-np.tanh(a1)**2)
    dW1 = -np.matmul(da1, np.transpose(x))
    db1 = -np.sum(da1, axis=1)
    db2 = -np.sum(e, axis=1)
    return dW1, dW2, db1[:,np.newaxis], db2[:,np.newaxis]

def dfa_backward_pass(e, h1, B1, a1, x):
    dW2 = -np.matmul(e, np.transpose(h1))
    da1 = np.matmul(B1, e)*(1-np.tanh(a1)**2)
    dW1 = -np.matmul(da1, np.transpose(x))
    db1 = -np.sum(da1, axis=1)
    db2 = -np.sum(e, axis=1)
    return dW1, dW2, db1[:,np.newaxis], db2[:,np.newaxis]

def average_angle(W2, B1, error, a1, a2):
    dh1 = np.mean(np.matmul(B1, error), axis=1)[:, np.newaxis] #forse non ci va la derivata
    c1 = np.mean(np.matmul(np.transpose(W2), error*(expit(a2)*(1-expit(a2)))), axis=1)[:, np.newaxis]
    dh1_norm = np.linalg.norm(dh1)
    c1_norm = np.linalg.norm(c1)
    inverse_dh1_norm = np.power(dh1_norm, -1)
    inverse_c1_norm = np.power(c1_norm, -1)
    
    # ALIGNMENT CRITERION AND ANGLE
    Lk = (np.matmul(np.transpose(dh1), c1)*inverse_dh1_norm)[0, 0]
    beta = np.arccos(np.clip(Lk*inverse_c1_norm, -1., 1.))*180/np.pi
    return Lk, beta

def train(x, y, n_epochs=10, lr=1e-3, batch_size=200, tol=1e-1):
    print("=== Train on BP ===")
    x = np.transpose(x)
    y = np.transpose(y)
    
    W1, W2 = np.random.randn(800, 784), np.random.randn(10, 800)
    b1, b2 = np.random.randn(800, 1), np.random.randn(10, 1)
    
    dataset_size = x.shape[1]
    n_batches = dataset_size//batch_size
    te_bp = []
    te_bp_std = []
    all_loss = []
    all_loss_std = []
    for i in range(n_epochs):
        perm = np.random.permutation(x.shape[1])
        x = x[:, perm]
        y = y[:, perm]
        loss = 0.
        train_error_lst = []
        train_error = 0.
        one_loss = []
        for j in range(n_batches):
            samples = x[:, j*batch_size:(j+1)*batch_size]
            targets = y[:, j*batch_size:(j+1)*batch_size]
            a1, h1, a2, y_hat = forward_pass(W1, W2, b1, b2, samples)
            error = y_hat - targets
            preds = np.argmax(y_hat, axis=0) 
            truth = np.argmax(targets, axis=0)
            train_error += np.sum(preds!=truth)
            loss_on_batch = log_loss(targets, y_hat)
            
            dW1, dW2, db1, db2 = backprop_backward_pass(error, h1, W2, a1, samples)
            W1 += lr*dW1
            W2 += lr*dW2
            b1 += lr*db1
            b2 += lr*db2
            loss += loss_on_batch
            one_loss.append(loss_on_batch)
            train_error_lst.append(np.sum(preds!=truth))
        training_error = 1.*train_error/x.shape[1]
        print('Loss at epoch', i+1, ':', loss/x.shape[1])
        print('Training error:', training_error)
        prev_training_error = 0 if i==0 else te_bp[-1]
        if np.abs(training_error-prev_training_error) <= tol:
            te_bp.append(training_error)
            break
        te_bp.append(training_error)
        te_bp_std.append(np.std(train_error_lst))

        all_loss.append(np.mean(one_loss))
        all_loss_std.append(np.std(one_loss))
    return W1, W2, b1, b2, te_bp, te_bp_std, all_loss, all_loss_std

def dfa_train(x, y, n_epochs=10, lr=1e-3, batch_size=200, tol=1e-3):
    print("=== Train on DFA ===")
    x = np.transpose(x)
    y = np.transpose(y)
    
    W1, W2 = np.random.randn(800, 784), np.random.randn(10, 800)
    b1, b2 = np.random.randn(800, 1), np.random.randn(10, 1)
    
    B1 = np.random.randn(800, 10)
    dataset_size = x.shape[1]
    n_batches = dataset_size//batch_size
    te_dfa = []
    te_dfa_std = []
    angles = []
    all_loss = []
    all_loss_std = []
    for i in range(n_epochs):
        perm = np.random.permutation(x.shape[1])
        x = x[:, perm]
        y = y[:, perm]
        loss = 0.
        train_error_lst = []
        train_error = 0.
        one_loss = []
        for j in range(n_batches):
            samples = x[:, j*batch_size:(j+1)*batch_size]
            targets = y[:, j*batch_size:(j+1)*batch_size]
            a1, h1, a2, y_hat = forward_pass(W1, W2, b1, b2, samples)
            error = y_hat - targets
            preds = np.argmax(y_hat, axis=0) 
            truth = np.argmax(targets, axis=0)
            train_error += 1.*np.sum(preds!=truth)
            loss_on_batch = log_loss(targets, y_hat)
            
            dW1, dW2, db1, db2 = dfa_backward_pass(error, h1, B1, a1, samples)
            W1 += lr*dW1
            W2 += lr*dW2
            b1 += lr*db1
            b2 += lr*db2
            loss += loss_on_batch
            one_loss.append(loss_on_batch)
            train_error_lst.append(1.*np.sum(preds!=truth))
            if j%100==0:
                angles.append(average_angle(W2, B1, error, a1, a2))
        training_error = 1.*train_error/x.shape[1]
        print('Loss at epoch', i+1, ':', loss/x.shape[1])
        print('Training error:', training_error)
        prev_training_error = 0 if i==0 else te_dfa[-1]
        if np.abs(training_error-prev_training_error) <= tol:
            te_dfa.append(training_error)
            break
        te_dfa.append(training_error)
        te_dfa_std.append(np.std(train_error_lst))

        all_loss.append(np.mean(one_loss))
        all_loss_std.append(np.std(one_loss))
    return W1, W2, b1, b2, te_dfa, te_dfa_std, angles, all_loss, all_loss_std

def test(W1, W2, b1, b2, test_samples, test_targets):
    test_samples = np.transpose(test_samples)
    test_targets = np.transpose(test_targets)
    outs = forward_pass(W1, W2, b1, b2, test_samples)[-1]
    preds = np.argmax(outs, axis=0) 
    truth = np.argmax(test_targets, axis=0)
    test_error = 1.*np.sum(preds!=truth)/preds.shape[0]
    return test_error

if __name__ == "__main__":
    np.random.seed(1234)

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    # print('Input dimensions')
    # print(X_train.shape, X_test.shape)
    # print(y_train.shape, y_test.shape)

    X_train = X_train.reshape(60000, 28*28)
    X_test = X_test.reshape(10000, 28*28)

    # print('After reshaping:', X_train.shape, X_test.shape)
    # print('Sample of label:', y_train[0])

    nb_classes = 10
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    # print('After conversion to categorical:', y_train[0])

    W1, W2, b1, b2, te_bp, te_bp_std, bp_loss, bp_loss_std = train(X_train, y_train, n_epochs=50, lr=1e-4, batch_size=32, tol=1e-4)
    W1dfa, W2dfa, b1dfa, b2dfa, te_dfa, te_dfa_std, angles, da_loss, da_loss_std = dfa_train(X_train, y_train, n_epochs=50, lr=1e-4, batch_size=32, tol=1e-4)

    print('BP:', test(W1, W2, b1, b2, X_test, y_test)*100, '%')
    print('DFA:', test(W1dfa, W2dfa, b1dfa, b2dfa, X_test, y_test)*100, '%')

    plt.figure()
    plt.errorbar(range(len(te_bp)), te_bp, yerr=te_bp_std, fmt='-o', ecolor='blue', capsize=5, label='BP training error')
    plt.errorbar(range(len(te_dfa)), te_dfa, yerr=te_dfa_std, fmt='-o', ecolor='red', capsize=5, label='DFA training error')
    plt.title('Training Error')
    plt.xlabel('Epochs')
    plt.ylabel('Training error')
    plt.legend(loc='best')
    plt.savefig('training.png')
    # plt.show()

    plt.figure()
    plt.errorbar(range(len(bp_loss)), bp_loss, yerr=bp_loss_std, fmt='-o', ecolor='blue', capsize=5, label='BP training error')
    plt.errorbar(range(len(da_loss)), da_loss, yerr=da_loss_std, fmt='-o', ecolor='red', capsize=5, label='DFA training error')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss %')
    plt.legend(loc='best')
    plt.savefig('error.png')

    def moving_average(data, window_size):
        """Compute the moving average of a 1D array with a specified window size."""
        window = np.ones(window_size) / window_size
        return np.convolve(data, window, 'valid')

    l, beta = zip(*angles)
    mvavg = moving_average(beta, 50)
    plt.figure()
    plt.plot(range(len(beta)), beta, label='angle')
    plt.plot(range(len(mvavg)), mvavg, label='angle movavg')
    plt.legend(loc='best')
    plt.xlabel('Counter')
    plt.ylabel('Angle')
    plt.savefig('angle.png')
    # plt.show()

