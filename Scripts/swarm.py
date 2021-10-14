import sys, pygame
import numpy as np
from math import *


# Swarm Motion
def swarm_motion(L, F, L_dir, F_dir, max_speed):

    '''
    :param L: Location of the Leaders
    :param F: Location of the followers
    :param L_dir: Speed and direction of the leader bots
    :param F_dir: Speed and direction of the follower bots
    :param max_speed: Max speed allowed for the bots

    :return F_dir: Output speed and direction of followers.
    '''

    rrep = 40
    rori = 50
    ratt = 70

    # Follower motion
    d = np.zeros((len(F), 1))
    bots = np.zeros(((len(F)+len(L)-1),2))
    bots[0:len(L),:] = L

    bots_dir = np.zeros(((len(F)+len(L)-1),2))
    bots_dir[0:len(L),:] = L_dir
    width,height = 750,750

    C1 = c2polar([width / 2, height / 2] - F)

    C2 = np.zeros(np.shape(C1))
    for i in range(0, len(C1)):
        if (C1[i, 0] - 160) > 0:
            C2[i, 0] = C1[i, 0] - 160
            C2[i, 1] = C1[i, 1]
        else:
            C2[i, 0] = 160 - C1[i, 0]
            C2[i, 1] = C1[i, 1] + 180
    C2 = c2polar(polar2c(C2))

    for i in range(len(F)):
        bots[len(L):, :] = F[(np.arange(len(F))!=i), :]
        bots_dir[len(L):, :] = F_dir[(np.arange(len(F)) != i), :]

        dir = c2polar(bots - F[i, :])

        # Orientation
        vecO = (rrep < dir[:, 0]) & (dir[:, 0] <= rori)
        checkO = np.where(vecO)
        idx = checkO[0]
        if idx.size != 0:
            ave_dir1 = bots_dir[idx, :]
            ex_pol = (ave_dir1.sum(axis=0) / idx.size)
            F_dir[i, :] = (F_dir[i, :] + ex_pol)/2

        # Attraction
        vecA = (rori< dir[:, 0]) & (dir[:, 0] <= ratt)
        vecA = np.logical_and(vecA, vecO)
        checkA = np.where(vecA)
        idx = checkA[0]
        if idx.size != 0:
            ave_dir1 = bots[idx, :] - F[i, :]
            expected_dir = (ave_dir1.sum(axis=0) / idx.size)
            ex_pol = c2polar(np.array([expected_dir]))
            ex_pol[0][0] = min(rrep - ex_pol[0][0], max_speed)
            F_dir[i, :] = (F_dir[i, :] + ex_pol) / 2

        #
        if C2[i,0]<rori:
            F_dir[i, 1] = (0.1 * F_dir[i, 1] + 0.9 * C2[i,1])

        # Repulsion
        vecR = dir[:, 0] <= rrep
        checkR = np.where(vecR)
        idx = checkR[0]
        if idx.size !=0:
            ave_dir1 = bots[idx, :]-F[i, :]
            expected_dir = (ave_dir1.sum(axis=0)/idx.size)
            ex_pol = c2polar(np.array([-expected_dir]))
            ex_pol[0][0] = min(rrep-ex_pol[0][0], max_speed)
            F_dir[i, :] = (0.1 * F_dir[i, :] +0.9 * ex_pol)



    return F_dir


def update_motion(L, F, L_dir, F_dir, max_speed):
    L_dir[:,0] = np.minimum(L_dir[:, 0], max_speed)
    F_dir[:,0] = np.minimum(F_dir[:, 0], max_speed)
    L = L + polar2c(L_dir)
    F = F + polar2c(F_dir)
    return L, F


# In the case if the bots hit the edges
def edge_case(x, dx, w, r):

    x1 = (x <= r)
    x1a = x1.astype(int)
    dx = dx * ((x1a * -2) + 1)
    x = (r * x1a) + (x * (1-x1a))

    x2 = (x >= (w - r))
    x2a = x2.astype(int)
    dx = dx * ((x2a * -2) + 1)
    x = ((w - r) * x2a) + (x * (1-x2a))

    return (x, dx)


def botcorrectedge(L, F, L_dir, F_dir, w, h, r):
    '''

    :param L: Location of the Leaders
    :param F: Location of the followers
    :param L_dir: Speed and direction of the leader bots
    :param F_dir: Speed and direction of the follower bots
    :param w: Width of the frame
    :param h: Height of the frame
    :param r: Radius of the bots

    :return L: Updated locations of the Leaders
    :return F: Updated locations of the followers
    :return L_dir: Updated speed and direction of the leader bots
    :return F_dir: Updated speed and direction of the follower bots

    '''
    # WALLS correction
    (L[:,0], L_dir[:,0]) = edge_case(L[:,0], L_dir[:,0], w, r)
    (L[:,1], L_dir[:,1]) = edge_case(L[:,1], L_dir[:,1], h, r)

    (F[:,0], F_dir[:,0]) = edge_case(F[:,0], F_dir[:,0], w, r)
    (F[:,1], F_dir[:,1]) = edge_case(F[:,1], F_dir[:,1], h, r)

    return (L, F, L_dir, F_dir)


def polar2c(Input):
    r = Input[:, 0]
    theta = Input[:, 1]
    x = r * np.cos((np.pi / 180) * theta)
    y = r * np.sin((np.pi / 180) * theta)
    Output = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
    return Output


def c2polar(Input):
    x = Input[:, 0]
    y = Input[:, 1]
    r = np.sqrt(x ** 2 + y ** 2)
    theta = (180 / np.pi) * np.arctan2(y, x)
    Output = np.concatenate((r.reshape((-1, 1)), theta.reshape((-1, 1))), axis=1)
    return Output


def euclidean_dist(a, b):
    return (sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def displaybots(L, F, width, height, r,screen):
    '''

    :param L: Location of the Leaders
    :param F: Location of the followers
    :param width: Width of the frame
    :param height: Height of the frame
    :param r: Radius of the bots
    :param screen: pygame object required for displaying the environment.
    '''

    # DISPLAY THE TARGET CIRCLE
    screen.fill((0, 0, 0))
    pygame.draw.circle(screen, (128, 0, 0), (int(width / 2), int(height / 2)), 165)
    pygame.draw.circle(screen, (0, 0, 0), (int(width / 2), int(height / 2)), 155)

    # DISPLAY LEADER BOTS
    for i in range(len(L)):
        pygame.draw.circle(screen, (0, 255, 0), (int(L[i, 0]), height - int(L[i, 1])), r)

    # DISPLAY FOLLOWER BOTS
    for i in range(len(F)):
        pygame.draw.circle(screen, (255, 255, 255), (int(F[i, 0]), height - int(F[i, 1])), r)

    pygame.display.update()
    pygame.time.delay(1)