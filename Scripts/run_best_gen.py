from main import Neuro_Evolution
import numpy as np
import scipy.io as sio


#cont = sio.whosmat("generations0.mat")
mat = sio.loadmat("generations1000.mat")


best_NN = mat['best_NN']


# Dimensions of the frame
size = width, height = 750, 750
frames = []

# Width of the bots
r = 8

# Total No of Bots
num = 30

# No of Leaders and Followers
num_leaders = 10
num_Followers = num - num_leaders

# Set max speed

max_speed = 2
size = width, height = 750, 750


# Repeate the simulation for N times
N = 20
Score = np.zeros((N,1))

# To View the simulation set display = 0
display = 0

for i in range(N):
    L = np.random.random_integers(10*r, 300, (num_leaders, 2))
    F = np.random.random_integers(10*r, 300, (num - num_leaders, 2))


    Score[i] = Neuro_Evolution(best_NN[0], L, F, max_speed, width, height, r, display)

print(str(np.mean(Score)))