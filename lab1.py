import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches

info = {
        # TODO replace the following with your own
        'Email' : 'anton.chumakov@polytechnique.edu',
        'Alias' : 'Antonchik', # optional
}

np.random.seed(1)

class Environment():

    def rwd(self, a, s): 
        '''
            Reward function r(a|s) of taking action a given state s

            Parameters
            ----------
            a : int
                action, 0 for non-attempt, or s to pounce to square s
            s : int
                state (true state; tile which containts the target)

            Returns
            -------
            int
                reward obtained from taking action a given state s
        '''
        if a == 0:
            # Non-attempt
            return 0
        elif s == a:
            # Capture
            return 10
        else:
            # Miss
            return -3

    def _dist(self,s1,s2):
        ''' return the l1 distance between tile s1 and tile s2 '''
        t1 = np.array([self._tile2coord(s1)])
        t2 = np.array([self._tile2coord(s2)])
        return np.sum(np.abs(t1 - t2))

    def _tile2coord(self,y):
        ''' convert tile number to coordinates (for plotting) '''
        return np.mod(y-1,self.n_cols),np.floor((y-1)/self.n_rows)

    def _tiles2path(self,path):
        ''' convert sequence of tiles to coordinates (for plotting) '''
        x_coords = []
        y_coords = []
        for tile in path:
            x_,y_ = self._tile2coord(tile)
            x_coords.append(x_)
            y_coords.append(y_)
        return x_coords, y_coords

    def _tile2cell(self,y):
        ''' convert tile number to matrix row, col (for math) '''
        return int(np.floor((y-1)/self.n_rows)),int(np.mod(y-1,self.n_cols))

    y_true = None

    def __init__(self,dims=[5,5]):
        ''' 
            Environment 
            -----------
        '''
        # Room shape
        self.n_cols = dims[0]
        self.n_rows = dims[1]
        self.G = np.zeros((self.n_cols,self.n_rows), dtype=int)
        self.n_states = self.n_cols * self.n_rows
        self.states = np.arange(1,self.n_states+1)
        # Room contents
        self.s_crinkle = [4,9,15,6,18,21,19,16,2,13]
        self.s_entries = [5,11]
        self.s_rustle = [1,7,17,19,25,9,16,2,13]
        self.s_object = 12
        for tile in self.s_crinkle:
            i,j = self._tile2cell(tile)
            self.G[i,j] = 1
        for tile in self.s_rustle:
            i,j = self._tile2cell(tile)
            if self.G[i,j] > 0:
                self.G[i,j] = 3
            else:
                self.G[i,j] = 2
        for tile in self.s_entries:
            i,j = self._tile2cell(tile)
            self.G[i,j] = 4
        # Room behaviour
        self.d_x = 2
        self.theta_0 = 0.9
        self.theta_1 = 0.8

    def P_Xy(self,s):
        ''' Distributions [P(X_1=1|y=s), P(X_2=1|y=s)].

            Parameters
            ----------
            s : int
                current tile/state

            Returns
            -------
                p where p[0] = p(X_1 = 1 | S = s), p[1] = p(X_2 = 1 | S = s)
        '''
        i,j = self._tile2cell(s)
        p = np.zeros(self.d_x)
        p[0] = self.theta_0 * (self.G[i,j] == 1 or self.G[i,j] == 3) 
        p[1] = self.theta_1 * (self.G[i,j] == 2 or self.G[i,j] == 3) 
        return p

    def P_Y(self):
        ''' Distribution P(Y[1]).

            Returns
            -------
            dict
                where d[s] = p(S = s)
        '''
        d = {}
        for s in self.s_entries:
            d[s] = 1/len(self.s_entries)
        return d

    def P_Yy(self,_y):
        ''' Distribution P(Y_t|y_{t-1}).

            Parameters
            ----------
            _y : int
                previous tile/state

            Returns
            -------
            dict
                where d[s] = P(Y_t = s | Y_{t-1} = _y).
        '''
        if _y is None:
            return self.P_Y()

        if _y not in self.states:
          raise Exception("Sorry, %d is not one of the states" % _y)

        d = {}

        for s in self.states:
            if self._dist(_y,s) == 1: 
                d[s] = 1

        # normalize
        Z = np.sum(np.array(list(d.values())))
        for s in d.keys():
            d[s] = d[s] / Z

        return d

    def gen_path(self, T=5):
        ''' Generate a path with associated observations.
            ---------------------------------------------

            Returns   
            -------
            X : A (T,self.d_x)-shape array of observations
            y : A T-length array of states
        '''

        x = np.zeros((T,self.d_x))
        y = np.zeros(T,dtype=int)

        # (t-1)-th state
        _y = None

        for t in range(T):

            # Generate a state
            d = self.P_Yy(_y)
            y_ = np.array(list(d.keys()))
            p_ = np.array(list(d.values()))/np.sum(np.array(list(d.values())))
            y[t] = np.random.choice(y_,p=p_)

            # Generate an observation
            P = self.P_Xy(y[t])
            x[t,:] = P > np.random.rand(self.d_x)

            # And remember this state
            _y = y[t]

        self.y_true = y
        return x

    def plot_scenario(self, path=None, dgrid=None, a_star=None):
        '''
            Plot visual representation of the environment.
            ---------------------------------------------
        '''

        fig, ax = plt.subplots(figsize=[8,4],dpi=300)

        # Agent starting position

        x,y = self._tile2coord(self.s_object)
        ax.plot(x,y,"bx",markersize=20,label="agent")

        # The tiles

        if dgrid is None:
            color_map = ListedColormap(["white", "green", "red", "orange", "yellow"])
            im = ax.imshow(self.G, cmap=color_map, interpolation='none',alpha=0.3)
        else:
            color_map = plt.cm.Reds
            im = ax.imshow(dgrid, cmap=color_map)

        # Pounce trajectory

        if a_star is not None and a_star > 0:
            t_x, t_y = self._tile2coord(a)
            ax.plot([x,t_x],[y,t_y],"b--",linewidth=2)

        # Path is an array of length T containing tile numbers, e.g., [1,3,1,2,...]

        if path is not None:
            T = len(path)
            x_coords,y_coords = self._tiles2path(path)
            ax.plot(x_coords+np.random.randn(T)*0.1,y_coords+np.random.randn(T)*0.1,"ro-")
            ax.plot(x_coords[-1],y_coords[-1],"rx",markersize=20)

        # Ticks and grid

        ax.set_xticks(np.arange(0, self.n_cols, 1))
        ax.set_xticks(np.arange(-0.5, self.n_cols, 1), minor=True)
        ax.set_xticklabels(np.arange(0, self.n_cols, 1))

        ax.set_yticks(np.arange(0, self.n_rows, 1))
        ax.set_yticks(np.arange(-0.5, self.n_rows, 1), minor=True)
        ax.set_xticklabels(np.arange(0, self.n_rows, 1))

        ax.grid(which='minor', color='k')

        n = 0
        for i in range(self.n_rows):
            for j in range(self.n_cols):
              ax.text(j, i, self.states[n], va='center', ha='center')
              n = n + 1

        # Legend (for the map)

        if dgrid is None:
            values = [1,2,3,4]
            labels = ['Crinkle', 'Rustle', 'Crinkle/rustle', 'Entry point']
            colors = [ im.cmap(im.norm(value)) for value in values]
            patches = [ mpatches.Patch(color=colors[i], alpha=0.3, label=labels[i] ) for i in range(len(values)) ]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

        # Return

        plt.tight_layout()
        return fig, ax


def path2str(y):
    ''' Convert list(int) into a string '''
    return str(y)

def str2path(y_str):
    ''' Convert string to a list(int) '''
    return [int(s) for s in y_str.split(' ')]


class Agent():

    # TODO: Add any more auxilliary functions you might need here 

    def P_Ho(self,x,env):
        '''
        Conditional joint distribution, P(Y_path | X_seq = x).

        Let

        y : list
            a *path*/state-sequence (list), e.g., [5,10,5,4,3]

        Note: Only need to include non-zero entries (any paths not included 
        are assumed to have 0 probability associated with them).

        Parameters
        ----------
        x : array_like(int,ndim=2)
            the T observations (2 bits each)
        env : Environment 
            the environment that produced observation x

        Returns
        -------
        dict(str:float)
            d such that d[str(y)] = P(Y_path = y | X_seq = x)
            if str(y) not in d, then we assume P(Y_path = y | X_seq = x) = 0.
        '''

        d = {}
        # TODO 
        P_Y = env.P_Y()

        # Iterate over all possible paths of length 5 using itertools.product
        for y in itertools.product(P_Y.keys(), repeat=5):
            # Compute the joint probability of the path and the observation sequence
            P_Yx = P_Y[y[0]]
            for i in range(1, 5):
                P_Yx *= self.PXY(x[i], y[i], env) * env.P_Yy(y[i-1])[y[i]]

            # Add the joint probability to the dictionary
            if P_Yx > 0.0:
                d[path2str(y)] = P_Yx

        # Normalize the probabilities by dividing each by the sum of all probabilities
        Z = sum(d.values())
        for s in d.keys():
            d[s] /= Z

        return d

    def P_Yo(self,d_joint):
        '''
        The (conditional) marginal distribution on final states, P(Y_T | X_seq = x).

        This is similar to the `predict_proba` function in scikit-learn.

        Let

        y : int
            the *final* state, e.g., 3

        Note: Only need to include non-zero entries (any states not included 
        are assumed to have 0 probability associated with them).

        Parameters
        ----------
        d_joint : dict(str:float)
            as returned from P_Tx

        Returns
        -------
        dict(int:float)
            d such that d[y] = P(Y_path = y | X_seq = x)
        '''
        
        d_marg = {}
        # TODO 
        # Iterate over all possible final states
        for y in self.env.Y:
            # Compute the marginal probability of the final state
            P_Yx = sum(d_joint[path] for path in d_joint.keys() if path[-1] == str(y))
            # Add the probability to the dictionary
            if P_Yx > 0.0:
                d_marg[y] = P_Yx

        # Normalize the probabilities by dividing each by the sum of all probabilities
        Z = sum(d_marg.values())
        for y in d_marg.keys():
            d_marg[y] /= Z

        return d_marg

    def Q_A(self,d_marginal,env):
        '''
            Expected reward E[r(S|A,x)] for any given action A = a.

            Note: Only need to include non-zero entries (any actions not 
            included are assumed to have 0 value associated with them).

            Parameters
            ----------
            d_marginal : dict(int:float)
                as returned from predict_proba
            env : Environment 
                the environment that implements env.rwd(a,s)

            Returns
            -------
            dict(int:float)
                d such that d[a] = E[r(S|a)]
        '''
        d_act = {}
        # TODO 
        # Iterate over all possible actions
        for a in env.A:
            # Compute the expected reward for the action
            E_rwd = sum(env.rwd(a, s) * d_marginal[s] for s in d_marginal.keys())
            # Add the expected reward to the dictionary
            if E_rwd > 0.0:
                d_act[a] = E_rwd

        return d_act

    def act(self,d_Q):
        '''
            Decide on an action to take, based on values Q 
            (take the best action). 

            Parameters
            ----------
            d_Q : dict(int:float)
                as returned from function Q_A

            Returns
            -------
            int,float
                the action a*, and its associated value Q(a*)
        '''
        if len(d_Q) <= 0:
            return 0,0
        # TODO 
        a_star = max(d_Q, key=d_Q.get)
        # Return the best action and its associated Q-value
        return a_star, d_Q[a_star]

def print_pmf(d,label="p(V = v) |  v"):
    ''' Pretty printing of dictionaries '''
    print(label)
    print("----------------------")
    for k in sorted(d, key=d.get, reverse=True):
        print("%6.3f    | %s" % (d[k],k))


if __name__ == "__main__":
    # -------------------------------------------------------------------- 
    # Main script
    # -------------------------------------------------------------------- 
    # Feel free to move this code into a jupyter notebook (if you prefer)
    # in which case, uncomment the following line:
    # from lab1 import Environment, Agent, print_pmf
    # 
    # You may make changes to this code as you wish.
    # --------------------------------------------------------------------

    # Init. the environment 
    env = Environment()
    print("Map: \n", env.G)
    x = env.gen_path()
    print("Observations: \n", x)
    fig, ax = env.plot_scenario()
    #plt.savefig('fig/scenario.pdf')
    plt.show()

    # Init. the agent 
    agent = Agent()

    # Task 1. Get joint-conditional distribution
    d_joint = agent.P_Ho(x,env)
    print("\nJoint distribution (of paths, for a given observation sequence): ")
    print_pmf(d_joint)

    # Task 2. Get marginal distribution
    d_marginal = agent.P_Yo(d_joint)
    print("\nMarginal distribution (of final state, for a given observation sequence): ")
    print_pmf(d_marginal)

    # Task 3. Make decision
    d_expected = agent.Q_A(d_marginal,env)
    print("\nExpected rewards (per action, for a given marginal distribution): ")
    print_pmf(d_expected,label="Q(a | s) |  a")
    a, exp_r = agent.act(d_expected)
    print("\nAction taken:", a, "; Expected reward:",exp_r)

    # Evaluation
    print("\nTrue path is:", env.y_true)
    r = env.rwd(a,env.y_true[-1])
    print("\nThe actual reward (from agent's decision):", r)
    # Plot the result
    P = np.zeros_like(env.G,dtype=float)
    for s in d_marginal.keys():
        i,j = env._tile2cell(s)
        P[i,j] = d_marginal[s]
    fig, ax = env.plot_scenario(path=env.y_true, dgrid=P, a_star=a)
    #plt.savefig('fig/result.pdf')
    plt.show()


