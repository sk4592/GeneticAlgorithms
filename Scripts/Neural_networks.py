from swarm import *


def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def FeedForwardNN(Input, w1, w2, b1, b2):
    '''

    :param Input: The values for the input layer.
    :param w1: Weights of the first layer.
    :param w2: weights of the second layer.
    :param b1: bias of the first layer.
    :param b2: bias of the second layer.
    :return: Output speed and direction of the leader.
    '''

    # NN

    h = np.matmul(Input, w1) +b1
    h_sig = np.arctan(h)
    y = np.matmul(h_sig, w2) + b2
    y_out = np.arctan(y)

    return y_out


def NN(NN_weights, L, F, max_speed, width, height):
    '''

    :param NN_weights: Weights of each Neural Network
    :param L: Location of the Leaders
    :param F: Location of the followers
    :param max_speed: Max speed allowed for the bots
    :param width: Width of the frame
    :param height: Height of the frame

    :return: Output speed and direction for leaders.
    '''
    # Total No of Bots
    num_leaders = len(L)
    w1 = NN_weights[0]
    w2 = NN_weights[1]
    b1 = NN_weights[2]
    b2 = NN_weights[3]

    Input1 = c2polar([width / 2, height / 2] - L)

    Input2 = np.zeros(np.shape(Input1))
    for i in range(0, len(Input1)):
        if (Input1[i, 0] - 160) > 0:
            Input2[i, 0] = Input1[i, 0] - 160
            Input2[i, 1] = Input1[i, 1]
        else:
            Input2[i, 0] = 160 - Input1[i, 0]
            Input2[i, 1] = Input1[i, 1] + 180
    Input2 = c2polar(polar2c(Input2))

    topF1 = np.zeros((len(L), 2))
    topF2 = np.zeros((len(L), 2))
    topF3 = np.zeros((len(L), 2))

    rsense = 70
    for i in range(len(L)):
        # Dist to top 3 followers
        dir_F = c2polar(F - L[i, :])
        idx = np.argsort(dir_F[:, 0])
        if dir_F[idx[0], 0] < rsense:
            topF1[i, :] = dir_F[idx[0], :]
        if dir_F[idx[1], 0] < rsense:
            topF2[i, :] = dir_F[idx[1], :]
        if dir_F[idx[2], 0] < rsense:
            topF3[i, :] = dir_F[idx[2], :]

    top_L = np.zeros((len(L), 2))
    for i in range(len(L)):
        # Dist to closest Leader
        dir_L = c2polar(L - L[i, :])
        idx = np.argsort(dir_L[:, 0])
        if dir_L[idx[1], 0] < rsense:
            top_L[i, :] = dir_L[idx[1], :]

    # pre-processing data
    Input = np.zeros((len(L), 10))
    scale = 50
    Input[:, 0] = ((np.minimum(Input1[:, 0], scale)) / scale)
    Input[:, 1] = ((np.minimum(topF1[:, 0], scale)) / scale)
    Input[:, 2] = ((np.minimum(topF2[:, 0], scale)) / scale)
    Input[:, 3] = ((np.minimum(topF3[:, 0], scale)) / scale)
    Input[:, 4] = ((np.minimum(top_L[:, 0], scale)) / scale)
    Input[:, 5] = Input1[:, 1]  / 180
    Input[:, 6] = topF1[:, 1] / 180
    Input[:, 7] = topF2[:, 1] / 180
    Input[:, 8] = topF3[:, 1] / 180
    Input[:, 9] = top_L[:, 1]  / 180

    Output = FeedForwardNN(Input, w1, w2, b1, b2)

    # post processing data
    L_dir = Output
    L_dir[:, 0] = (Output[:, 0]+np.pi)/(2*np.pi) * max_speed
    L_dir[:, 1] = (180/np.pi) * (Output[:, 1])

    d_temp = c2polar([width / 2, height / 2] - F)
    d1 = np.absolute(d_temp[:, 0])
    d = np.absolute(d1 - 160)

    score = gaussian(d, 0, 60)

    score_ave = (score.sum() / len(F))
    return L_dir, score_ave