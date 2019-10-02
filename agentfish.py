######################################################################################################################
# HUJI 67800 - Probabilistic Methods in AI
# Reinforcement Learning Programming Assignment
#     ******** The Fish Pond ********
######################################################################################################################
import numpy as np
from fishpond import FishPond
from matplotlib import pyplot as plt

actions = ['l', 'u', 'd', 'r']
numed_actions = {'l': 0, 'u': 1, 'd': 2, 'r': 3}


def the_right_policy():
    return 'r'


def q0_sample_policy(pond: FishPond):
    # Sample code for running a policy and plotting the trajectory
    for i in range(30):
        action = the_right_policy()
        reached_end = pond.perform_action(action)
        pond.plot()
        if reached_end:
            break
    print('Done')
    plt.savefig('Q0_' + pond_name + '.png')
    plt.show()


def the_greedy_policy(current_state, end_state):
    """chooses a greedy move as a function of the current location of the fish in the pond"""
    options = []
    if current_state[0] < end_state[0]:
        options += ['r']
    elif current_state[0] > end_state[0]:
        options += ['l']
    if current_state[1] < end_state[1]:
        options += ['u']
    elif current_state[1] > end_state[1]:
        options += ['d']
    return np.random.choice(options)


def q1_greedy_policy(pond: FishPond):
    """given a fishpond, runs a game with respect to the greedy policy as implemented in the function
    the_greedy_policy"""
    path_len = np.abs(pond.start_state[0] - pond.end_state[0]) + np.abs(pond.start_state[1] - pond.end_state[1])
    # outer loop for number of episodes
    for e in range(1):
        pond.reset()
        # inner loop for an episode
        for i in range(3 * path_len):
            action = the_greedy_policy(pond.current_state, pond.end_state)
            reached_end = pond.perform_action(action)
            pond.plot()
            if reached_end:
                break
    print('Done')
    plt.savefig('Q1_' + pond_name + '.png')
    plt.show()


def q2_compute_Q(prev_Q, pond: FishPond, gamma):
    """ computes the Q values for each state and action at a given iteration"""
    r_s_a = -1
    Q = np.zeros(prev_Q.shape)
    Q_max = np.max(prev_Q, axis=2)
    for x in range(pond.pond_size[0]):
        for y in range(pond.pond_size[1]):
            add_reward = 0 if x == pond.end_state[0] and y == pond.end_state[1] else 1
            for i, a in enumerate(actions):
                p_s_tag = pond.get_action_outcomes((x, y), a)
                p_tag_summed = 0
                for option in p_s_tag:
                    pos = option[1]
                    add_reward = 0 if pos[0] == pond.end_state[0] and pos[1] == pond.end_state[1] else add_reward
                    p_tag_summed += Q_max[pos[0], pos[1]] * option[0]
                p_tag_summed *= gamma
                Q[x, y, i] = p_tag_summed
                if add_reward:
                    Q[x, y, i] += r_s_a
    return Q


def q2_learn_q_phi(pond: FishPond, gamma):
    """policy iteration implementation for Q2"""
    prev_pi = np.zeros(pond.pond_size)
    Q = np.zeros((pond.pond_size[0], pond.pond_size[1], 4))
    Q[pond.end_state[0], pond.end_state[1]] = 0
    pi = np.random.choice(np.arange(4), size=pond.pond_size)
    while not np.allclose(prev_pi, pi, rtol=1e-5):
        prev_pi = pi
        Q = q2_compute_Q(Q, pond, gamma)
        pi = np.argmax(Q, axis=2)
    return pi, Q


def q2_greedy_policy(pond: FishPond, gamma):
    """given a fishpond, runs a single game with the policy computed with the policy iteration procedure implemented
    in the function q2_learn_q_phi"""
    path_len = np.abs(pond.start_state[0] - pond.end_state[0]) + np.abs(pond.start_state[1] - pond.end_state[1])
    pi, _ = q2_learn_q_phi(pond, gamma)
    pi = pi.astype(np.int64)
    pond.reset()
    for i in range(3 * path_len):
        action = pi[pond.current_state[0], pond.current_state[1]]
        action = actions[action]
        reached_end = pond.perform_action(action)
        pond.plot()
        if reached_end:
            break
    print('Done')
    plt.savefig('Q2_' + pond_name + '.png')
    plt.show()


def q3_learn_off(gamma, alpha, ep_num, ep_len, eps, pond: FishPond, Q_asterisk):
    """
    creates episodes for Q - learning with action choosing with greedy policy
    :param ep_num: the number of episodes to create
    :param ep_len: the length of each episode to create
    :param eps: epsilon rate for the greedy policy
    """
    Q = np.zeros((pond.pond_size[0], pond.pond_size[1], 4))
    # Q = np.ones((pond.pond_size[0], pond.pond_size[1], 4)) * (-1)
    # Q[pond.end_state[0], pond.end_state[1]] = 0
    err = np.empty(0)
    for e in range(ep_num):
        pond.reset()
        print(e)
        i = 0
        for j in range(ep_len):
            # while not reached_end:
            x, y = pond.current_state
            a1 = the_greedy_policy(pond.current_state, pond.end_state)
            a2 = np.random.choice(actions)
            action = np.random.choice([a1, a2], 1, p=[1 - eps, eps])[0]
            n_a = numed_actions[action]
            reached_end = pond.perform_action(action)
            if reached_end:
                r_s_a = 0
            else:
                r_s_a = -1
            x_tag, y_tag = pond.current_state
            Q[x, y, n_a] = Q[x, y, n_a] + alpha * (r_s_a + gamma * np.max(Q[x_tag, y_tag, :]) - Q[x, y, n_a])
            i += 1
            if reached_end:
                break
        err = np.append(err, get_MSE(Q, Q_asterisk))
    return err, Q


def q3_learn_on(gamma, alpha, ep_num, ep_len, eps, pond: FishPond, Q_asterisk):
    """
    creates episodes for Q - learning with action choosing using the learned Q
    :param ep_num: the number of episodes to create
    :param ep_len: the length of each episode to create
    :param eps: epsilon rate for the greedy policy
    :param Q_asterisk: optimal Q
    """
    err, Q = q3_learn_off(gamma, alpha, 1, ep_len, eps, pond, Q_asterisk)
    for e in range(ep_num - 1):
        pond.reset()
        print(e)
        for j in range(ep_len):
            x, y = pond.current_state
            n_a = np.argmax(Q[x, y, :])
            action = actions[n_a]
            reached_end = pond.perform_action(action)
            if reached_end:
                r_s_a = 0
            else:
                r_s_a = -1
            x_tag, y_tag = pond.current_state
            # temp_Q[x, y, n_a] = Q[x, y, n_a] + alpha * (r_s_a + gamma * np.max(Q[x, y, :]) - Q[x, y, n_a])
            Q[x, y, n_a] = Q[x, y, n_a] + alpha * (r_s_a + gamma * np.max(Q[x_tag, y_tag, :]) - Q[x, y, n_a])
            if reached_end:
                break
        err = np.append(err, get_MSE(Q, Q_asterisk))
    return err, Q


def get_MSE(Q1, Q2):
    """computes the MSE between two matrices"""
    err = (Q1 - Q2) ** 2
    return np.mean(err)


def plot_errors(err_off, err_on):
    """ plot MSE values of 3a and 3b settings as a function of the number of episodes the algorithm learned """
    line = np.arange(err_on.shape[0]) + 1
    plt.plot(line, err_off, c='lightblue', linewidth=2, label='3a')
    plt.plot(line, err_on, c='salmon', linewidth=2, label='3b')
    plt.autoscale(enable=True, axis=u'both', tight=False)
    plt.ylabel('MSE')
    plt.xlabel('episode number')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('Q3_errors_' + pond_name + '.png')
    plt.show()


def q3_play_a_game(pond: FishPond, pi, section):
    path_len = np.abs(pond.start_state[0] - pond.end_state[0]) + np.abs(pond.start_state[1] - pond.end_state[1])
    pond.reset()
    for i in range(3 * path_len):
        action = pi[pond.current_state[0], pond.current_state[1]]
        action = actions[action]
        reached_end = pond.perform_action(action)
        pond.plot()
        if reached_end:
            break
    print('Done')
    plt.savefig('Q3_' + pond_name + '_' + section + '_game.png')
    plt.show()


def Q3(pond: FishPond):
    """wrapper function for Q3. runs Q-learning for each of the settings, then plots the errors"""
    gamma = 0.45
    alpha = 0.1
    epsilon = 0.5
    pond.reset()
    path_len = np.abs(pond.start_state[0] - pond.end_state[0]) + np.abs(pond.start_state[1] - pond.end_state[1])
    pi, Q_asterisk = q2_learn_q_phi(pond, gamma)
    err_off, Q_off = q3_learn_off(gamma, alpha, 30000, 3 * path_len, epsilon, pond, Q_asterisk)
    err_on, Q_on = q3_learn_on(gamma, alpha, 30000, 3 * path_len, epsilon, pond, Q_asterisk)
    plot_errors(err_off, err_on)

    # play a game
    pi_off = np.argmax(Q_off, axis=2)
    pi_on = np.argmax(Q_on, axis=2)
    q3_play_a_game(pond, pi_off, 'a')
    q3_play_a_game(pond, pi_on, 'b')


if __name__ == "__main__":
    names = ['pond1', 'pond2', 'pond3', 'pond4', 'pond43', 'pond5']
    for pond_name in names:
        my_pond = FishPond(pond_name + '.txt')
        # Run some sample policy for reference
        q0_sample_policy(my_pond)
        q1_greedy_policy(my_pond)
        q2_greedy_policy(my_pond, 1)
        Q3(my_pond)
