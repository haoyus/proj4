from mdptoolbox import example, mdp
from matplotlib import pyplot as plt
import numpy as np
import random as rand

def testForest():
    n_iters_vi = []
    times_vi = []

    n_iters_pi = []
    times_pi = []

    policy_compare = []
    policy_compare_q = []
    sizes = []
    disc = 0.9
    for s in range(3,101):
        sizes.append(s)
        P, R = example.forest(S=s, p=0.1)
        print(P)
        print(R)

        vi = mdp.ValueIteration(P,R,
            discount=disc,
            epsilon=0.01,
            max_iter=1000
        )
        vi.run()
        print('vi policy: ',vi.policy)
        print('vi V: ',vi.V)
        # print(vi.iter)
        # print(vi.time)
        n_iters_vi.append(vi.iter)
        times_vi.append(vi.time*1000.)

        pi = mdp.PolicyIteration(P, R,
            discount=disc,
            max_iter=1000
        )
        pi.run()
        n_iters_pi.append(pi.iter)
        times_pi.append(pi.time*1000.)

        ql = mdp.QLearning(P, R, disc)
        ql.run()
        print('ql policy: ', ql.policy)
        print('ql V: ', ql.V)
        
        pol_vi = np.array(vi.policy)
        pol_pi = np.array(pi.policy)
        pol_ql = np.array(ql.policy)

        pol_comp = (pol_vi==pol_pi)
        pol_comp = int(np.all(pol_comp))
        policy_compare.append(pol_comp)

        pol_comp_q = int(np.all(pol_vi==pol_ql))
        policy_compare_q.append(pol_comp_q)

    return
    # print(len(pi.V))
    creatPlot('Forest Management', 'problem size', n_iters_vi, n_iters_pi, times_vi, times_pi, policy_compare,policy_compare_q, sizes)

def creatPlot(test_name, x_axis_name, n_iters_vi, n_iters_pi, times_vi, times_pi, policy_compare,policy_compare_q, sizes):
    fig, ax = plt.subplots(3,1, figsize=(4,8))
    fig.suptitle(f'Val iteration vs Policy iteration on {test_name}')
    ax[0].plot(sizes, n_iters_vi, color='b', label='val iter')
    ax[0].plot(sizes, n_iters_pi,      'g.', label='pol iter')
    ax[0].grid()
    ax[0].legend()
    ax[0].set_xlabel(x_axis_name)
    ax[0].set_ylabel('converge iters')

    ax[1].plot(sizes, times_vi, color='b', label='val iter')
    ax[1].plot(sizes, times_pi, 'g.',      label='pol iter')
    ax[1].grid()
    ax[1].legend()
    ax[1].set_xlabel(x_axis_name)
    ax[1].set_ylabel('converge time (ms)')

    ax[2].plot(sizes, policy_compare, 'y.',      label='if policies are same')
    ax[2].grid()
    ax[2].legend()
    ax[2].set_xlabel(x_axis_name)
    ax[2].set_ylabel('policy comparison')
    
    fig.tight_layout()
    plt.savefig(f'{test_name}_iter_time.png')
    plt.clf()

    fig, ax = plt.subplots(1,1, figsize=(4,3))
    fig.suptitle('Q Learning policy vs MDP')
    ax.plot(sizes, policy_compare_q,      'g.', label='if policies are same')
    ax.grid()
    ax.legend()
    ax.set_xlabel(x_axis_name)
    ax.set_ylabel('policy comparison')
    fig.tight_layout()
    plt.savefig(f'{test_name} qlearning policy.png')
    plt.clf()

##################################################################

def load_map():
    verbose = False
    filename = 'world.csv'
    inf = open(filename)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    data = np.array(  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        [list(map(float, s.strip().split(","))) for s in inf.readlines()]  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    )  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    originalmap = (  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        data.copy()  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    )  # make a copy so we can revert to the original map later

    # tmp=''
    # for s in inf.readlines():
    #     s=s.strip()
    #     s=list(map(float, s.split(',')))
    #     # s=s.strip('\n')
    #     # print(s)
    if verbose:  	
        print(data)
        print('------')	  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        printmap(data)
    return data

def printmap(data):  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    Prints out the map  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param data: 2D array that stores the map  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type data: array  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print("--------------------")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    for row in range(0, data.shape[0]):  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        for col in range(0, data.shape[1]):  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
            if data[row, col] == 0:  # Empty space  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
                print(" ", end=" ")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
            if data[row, col] == 1:  # Obstacle  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
                print("O", end=" ")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
            if data[row, col] == 2:  # El roboto  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
                print("*", end=" ")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
            if data[row, col] == 3:  # Goal  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
                print("X", end=" ")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
            if data[row, col] == 4:  # Trail  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
                print(".", end=" ")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
            if data[row, col] == 5:  # Quick sand  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
                print("~", end=" ")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
            if data[row, col] == 6:  # Stepped in quicksand  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
                print("@", end=" ")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        print()  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print("--------------------") 

def try_move(r,c,a):
    if a == 0:  # north  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        r = r - 1
    elif a == 1:  # east  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        c = c + 1
    elif a == 2:  # south  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        r = r + 1
    elif a == 3:  # west
        c = c - 1
    return r,c
def bound(r_old,c_old, r,c, mapdata):
    if r<0 or r>=20 or c<0 or c>=20:
        return r_old, c_old, False
    elif mapdata[r,c]==0 or mapdata[r,c]==5 or mapdata[r,c]==2 or mapdata[r,c]==3:
        return r, c, True
    elif mapdata[r,c]==1:
        return r_old, c_old, False
    else:
        raise Exception('wrong move')
def move(r, c, a, mapdata):
    r_old, c_old = r, c
    # p = rand.uniform(0.0, 1.0)
    p_left = 0.1
    a_left = a - 1 if a>0 else 3
    p_straight = 0.8
    a_straight = a
    p_right = 0.1
    a_right = a + 1 if a<3 else 0

    p_stay = 0
    
    # left
    r_l, c_l = try_move(r, c, a_left)
    r_l, c_l, movable_l = bound(r,c,r_l, c_l, mapdata)
    # right
    r_r, c_r = try_move(r, c, a_right)
    r_r, c_r, movable_r = bound(r,c,r_r, c_r, mapdata)
    # straight
    r_s, c_s = try_move(r, c, a_straight)
    r_s, c_s, movable_s = bound(r,c,r_s, c_s, mapdata)

    ret = []
    if movable_l:
        ret.append((20*r_l+c_l, p_left))
    else:
        p_stay += p_left
    if movable_r:
        ret.append((20*r_r+c_r, p_right))
    else:
        p_stay += p_right
    if movable_s:
        ret.append((20*r_s+c_s, p_straight))
    else:
        p_stay += p_straight
    ret.append((20*r+c, p_stay))

    return ret


def get_transition():
    mapdata = load_map()
    P = np.zeros((4,400,400))
    R = np.zeros((400,4))
    for i in range(20):
        for j in range(20):
            s_curr = i*20 + j
            for a in range(4):
                updates = move(i, j, a, mapdata)
                for s_next, p in updates:
                    P[a,s_curr,s_next] = p
            if mapdata[i,j]==0 or mapdata[i,j]==2 or mapdata[i,j]==1:
                R[s_curr,:] = -1
            elif mapdata[i,j]==5:
                R[s_curr,:] = -100
            elif mapdata[i,j]==3:
                R[s_curr,:] = 500

    return P, R

def testMaze():
    P, R = get_transition()
    
    n_iters_vi = []
    times_vi = []

    n_iters_pi = []
    times_pi = []

    policy_compare = []
    policy_compare_q = []
    discs = []

    for d in range(60,100,2):
        disc = d / 100.
        discs.append(disc)

        vi = mdp.ValueIteration(P,R,
            discount=disc,
            epsilon=0.01,
            max_iter=1000
        )
        vi.run()
        # print('value iteration policy:',vi.policy)
        # print('value iteration Value function:',vi.V)
        # print(vi.iter)
        # print(vi.time)
        n_iters_vi.append(vi.iter)
        times_vi.append(vi.time*1000.)

        pi = mdp.PolicyIteration(P, R,
            discount=disc,
            max_iter=1000
        )
        pi.run()
        # n_iters_pi.append(pi.iter)
        # times_pi.append(pi.time*1000.)
        # print('policy iteration policy:',vi.policy)
        # print('policy iteration Value function:',vi.V)
        n_iters_pi.append(pi.iter)
        times_pi.append(pi.time*1000.)

        ql = mdp.QLearning(P, R, disc)
        ql.run()
        # print('Q learning policy: ', ql.policy)
        # print('Q learning Value function: ', ql.V)

        pol_vi = np.array(vi.policy)
        pol_pi = np.array(pi.policy)
        pol_ql = np.array(ql.policy)

        pol_comp = (pol_vi==pol_pi)
        pol_comp = int(np.all(pol_comp))
        policy_compare.append(pol_comp)

        pol_comp_q = int(np.all(pol_vi==pol_ql))
        policy_compare_q.append(pol_comp_q)

    creatPlot('Maze', 'discount rate', n_iters_vi, n_iters_pi, times_vi, times_pi, policy_compare,policy_compare_q, discs)



if __name__ == '__main__':
    testForest()
    testMaze()