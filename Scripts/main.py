from Neural_networks import *
from joblib import Parallel, delayed, parallel_backend
import multiprocessing
import time
import scipy.io as sio


def Neuro_Evolution(NN_weights, L, F, max_speed, width, height, r, display):

    '''
    :param NN_weights: Weights of each Neural Network
    :param L: Location of the Leaders
    :param F: Location of the followers
    :param max_speed: Max speed allowed for the bots
    :param width: Width of the frame
    :param height: Height of the frame
    :param r: Radius of the bots
    :param display: Set display = 0 to see the simulation

    :return overall_score: The output of the fitness function called Score is returned.
    '''


    size = width, height
    if display == 0:
        screen = pygame.display.set_mode(size)
        pygame.display.init()

    num_followers = len(F)

    tau = 500
    score = np.zeros((tau, 1))

    w = 1 - np.cos(np.pi * np.linspace(0, 1, num=tau))

    num_leaders = len(L)
    # Leader motion
    L_dir = np.zeros((num_leaders, 2))
    uni_dist = np.random.uniform(size=(num_leaders, 2))
    L_dir[:, 0] = (max_speed / 2) * uni_dist[:, 0]  # neu
    L_dir[:, 1] = (2 * 180) * uni_dist[:, 1] - 180  # phi

    num_followers = len(F)
    # Followers motion
    F_dir = np.zeros((num_followers, 2))
    uni_dist = np.random.uniform(size=(num_followers, 2))
    F_dir[:, 0] = (0) * uni_dist[:, 0]  # neu
    F_dir[:, 1] = (2 * 180) * uni_dist[:, 1] - 180  # phi

    count = 0
    for i in range(tau):
        if display == 0:
            count += 1
            displaybots(L, F, width, height, r, screen)
            #pygame.image.save(screen, "bots" +str(count)+ ".png")


            # Save into a GIF file that loops forever


        L_dir1, score[i] = NN(NN_weights, L, F, max_speed, width, height)
        L_dir = (L_dir+L_dir1)/2
        F_dir = swarm_motion(L, F, L_dir, F_dir, max_speed)
        L, F = update_motion(L, F, L_dir, F_dir, max_speed)
        (L, F, L_dir, F_dir) = botcorrectedge(L, F, L_dir, F_dir, width, height, r)


    overall_score = (1 / tau) * np.matmul(w, score)

    return overall_score


def mutate(dict_NN,score):
    '''

    :param dict_NN: All the Neural networks spanning the entire population.
    :param score: The Score value associate with each network.

    :return dict_NN_new: The New population after mutation.

    '''

    # Pick the top 10 performers
    idx = np.argsort(score)[::-1]
    k = 0
    dict_NN_new = {}

    No_top_NN = 10
    # Perform Mutation and replace the population with newly mutated Neural Networks
    for i in range(0, No_top_NN):
        dict_NN_top = dict_NN[int(idx[i])]

        dict_NN_new[k] = dict_NN_top
        k += 1

        # Mutate the top performers

        # Mutator settings
        NN_mutate = 0.2
        mu, sigma = 0, NN_mutate
        input_nodes = 10
        hidden_nodes = 16
        output_nodes = 2

        # First Mutation
        w1 = dict_NN_top[0] + np.random.normal(mu, sigma, (input_nodes, hidden_nodes))
        w2 = dict_NN_top[1] + np.random.normal(mu, sigma, (hidden_nodes, output_nodes))
        b1 = dict_NN_top[2] + np.random.uniform(-sigma*np.pi, sigma, (1, hidden_nodes))
        b2 = dict_NN_top[3] + np.random.uniform(-sigma*np.pi, sigma, (1, output_nodes))

        dict_NN_mutate1 = np.array([w1, w2, b1, b2])
        dict_NN_new[k] = dict_NN_mutate1
        k += 1

        # Second Mutation
        w1 = dict_NN_top[0] + np.random.normal(mu, sigma, (input_nodes, hidden_nodes))
        w2 = dict_NN_top[1] + np.random.normal(mu, sigma, (hidden_nodes, output_nodes))
        b1 = dict_NN_top[2] + np.random.uniform(-sigma, sigma, (1, hidden_nodes))
        b2 = dict_NN_top[3] + np.random.uniform(-sigma, sigma, (1, output_nodes))

        dict_NN_mutate2 = np.array([w1, w2, b1, b2])
        dict_NN_new[k] = dict_NN_mutate2
        k += 1

    return dict_NN_new

# Main function

if __name__ == "__main__":
    # Dimensions of the frame
    size = width, height = 750, 750
    frames = []

    # Width of the bots
    r = 8

    # Total No of Bots
    num = 30

    # No of Leaders and Followers
    num_leaders = 12
    num_Followers = num - num_leaders

    # Set max speed
    max_speed = 2

    # Creating 15 neural networks
    dict_NN = {}
    No_NN = 30
    for i in range(0,No_NN):
        mu, sigma = 0, 1
        input_nodes = 10
        hidden_nodes = 16
        output_nodes = 2
        w1 = np.random.normal(mu, sigma, (input_nodes, hidden_nodes))
        w2 = np.random.normal(mu, sigma, (hidden_nodes, output_nodes))
        b1 = np.random.uniform(-np.pi, np.pi, (1, hidden_nodes))
        b2 = np.random.uniform(-np.pi, np.pi, (1, output_nodes))
        dict_NN[i] = np.array([w1,w2,b1,b2])



    # Run for 1000 Generations
    No_Generations = 1000
    score = np.zeros((No_NN,1))

    num_cores = multiprocessing.cpu_count()
    track_E = np.zeros((No_NN,No_Generations))

    for i in range(0,No_Generations):
        L = np.random.random_integers(10*r, 300, (num_leaders, 2))
        F = np.random.random_integers(10*r, 300, (num - num_leaders, 2))


        Start = time.time()
        score = np.array(Parallel(n_jobs=num_cores)(delayed(Neuro_Evolution)(dict_NN[j], L, F, max_speed, width, height, r, j) for j in range(0, No_NN)))
        Finish = time.time()

        print('Generation Number: '+str(i)+', time: '+ str(Finish - Start), ', Score: ' + str(score[0]))
        best_ind = np.argmax(score)
        best_NN = dict_NN[best_ind]
        sio.savemat("generations" + str(i) + ".mat", {'best_NN': best_NN})

        track_E[:, i] = score[:, 0]
        dict_NN = mutate(dict_NN, score)


    None