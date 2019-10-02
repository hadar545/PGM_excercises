###############################################################################
# factor graph data structure implementation 
# author: Ya Le, Billy Jun, Xiaocheng Li
# date: Jan 25, 2018
###############################################################################

import functools
import numpy as np
from factors import *
import scipy.spatial.distance as dista


class FactorGraph:
    def __init__(self, numVar=0, numFactor=0):
        '''
        var list: index/names of variables

        domain list: the i-th element represents the domain of the i-th variable; 
                     for this programming assignments, all the domains are [0,1]

        varToFactor: list of lists, it has the same length as the number of variables. 
                     varToFactor[i] is a list of the indices of Factors that are connected to variable i

        factorToVar: list of lists, it has the same length as the number of factors. 
                     factorToVar[i] is a list of the indices of Variables that are connected to factor i

        factors: a list of Factors

        messagesVarToFactor: a dictionary to store the messages from variables to factors,
                            keys are (src, dst), values are the corresponding messages of type Factor

        messagesFactorToVar: a dictionary to store the messages from factors to variables,
                            keys are (src, dst), values are the corresponding messages of type Factor
        '''

        self.var = [None for _ in range(numVar)]
        self.domain = [[0, 1] for _ in range(numVar)]
        self.varToFactor = [[] for _ in range(numVar)]
        self.factorToVar = [[] for _ in range(numFactor)]
        self.factors = []
        self.messagesVarToFactor = {}
        self.messagesFactorToVar = {}

    def addFactor(self, factor):
        '''
        :param factor: a Factor object
        '''
        self.factors.append(Factor(factor))
        assert len(self.factors) <= len(self.factorToVar)
        for var_idx in factor.scope:
            self.varToFactor[var_idx].append(len(self.factors) - 1)
        self.factorToVar[len(self.factors) - 1] = factor.scope

    def evaluateWeight(self, assignment):
        '''
        param - assignment: the full assignment of all the variables
        return: the multiplication of all the factors' values for this assignment
        '''
        a = np.array(assignment, copy=False)
        output = 1.0
        for f in self.factors:
            output *= f.val[tuple(a[f.scope])]
        return output

    def getInMessage(self, src, dst, type="varToFactor"):
        '''
        param - src: the source factor/clique index
        param - dst: the destination factor/clique index
        param - type: type of messages. "varToFactor" is the messages from variables to factors; 
                    "factorToVar" is the message from factors to variables
        return: message from src to dst
        
        In this function, the message will be initialized as an all-one vector (normalized) if 
        it is not computed and used before. 
        '''
        if type == "varToFactor":
            if (src, dst) not in self.messagesVarToFactor:
                inMsg = Factor()
                inMsg.scope = [src]
                inMsg.card = [len(self.domain[src])]
                inMsg.val = np.ones(inMsg.card) / inMsg.card[0]
                self.messagesVarToFactor[(src, dst)] = inMsg
            return self.messagesVarToFactor[(src, dst)]

        if type == "factorToVar":
            if (src, dst) not in self.messagesFactorToVar:
                inMsg = Factor()
                inMsg.scope = [dst]
                inMsg.card = [len(self.domain[dst])]
                inMsg.val = np.ones(inMsg.card) / inMsg.card[0]
                self.messagesFactorToVar[(src, dst)] = inMsg
            return self.messagesFactorToVar[(src, dst)]

    def runParallelLoopyBP(self, iterations, hamming=False, ping_locs=np.empty(0)):
        '''
        param - iterations: the number of iterations you do loopy BP

        In this method, you need to implement the loopy BP algorithm. The only values
        you should update in this function are self.messagesVarToFactor and self.messagesFactorToVar.

        Warning: Don't forget to normalize the message at each time. You may find the normalize
        method in Factor useful.

        Note: You can also calculate the marginal MAPs after each iteration here...
        '''
        ping_output = np.empty((len(self.var), ping_locs.shape[0]))
        hamm_log = []
        mult_ufunc = np.frompyfunc(self.mult_factors, 2, 1)
        # initialize the varToFactor & FactorToVar messages and factors
        for fa in self.factors:
            fa_scope = fa.scope
            for i, var in enumerate(fa_scope):
                self.getInMessage(var, fa, type="varToFactor")
        for it in range(iterations):
            print('.', end='', flush=True)
            if (it + 1) % 5 == 0:
                print(it + 1, end='', flush=True)

            # handle messages from factors to vars
            for fa in self.factors:
                fa_scope = fa.scope
                # handling unary factors
                if len(fa_scope) == 1:
                    self.messagesFactorToVar[(fa, fa_scope[0])] = fa.normalize()
                    continue
                # handling multiple-var-factors
                relv_msgs = self.getRelvMessages(fa, fa_scope)
                final_msgs = [mult_ufunc.reduce(relv_msgs[:i] + relv_msgs[i + 1:]) for i in range(len(relv_msgs))]
                final_msgs = [self.mult_factors(fa, final_msgs[i]) for i in range(len(final_msgs))]
                final_msgs = [fa.marginalize_all_but([fa_scope[i]]).normalize() for i, fa in enumerate(final_msgs)]
                for i, v in enumerate(fa_scope):
                    self.messagesFactorToVar[(fa, v)] = final_msgs[i]
            # handle messages from vars to factors
            for var in self.var:
                relv_fa_locs = self.varToFactor[var]
                relv_fas = [self.factors[i] for i in relv_fa_locs]
                relv_msgs = [self.messagesFactorToVar[(r_fa, var)] for r_fa in relv_fas]
                final_msgs = [mult_ufunc.reduce(relv_msgs[:i] + relv_msgs[i + 1:]) for i in range(len(relv_msgs))]
                for i, fa in enumerate(relv_fas):
                    self.messagesVarToFactor[(var, fa)] = final_msgs[i]
            if hamming:
                hamm_log += [self.getHamming()]
            if ping_locs.shape[0] and it in ping_locs:
                curr_loc = np.where(it == ping_locs)[0][0]
                ping_output[:, curr_loc] = self.getMarginalMAP()
        if hamming:
            return hamm_log
        if ping_output.shape[0]:
            return ping_output

    @staticmethod
    def mult_factors(fac1, fac2):
        """multiplies two factors whilst keeping the probability function valid"""
        return fac1.multiply(fac2).normalize()

    def getRelvMessages(self, fa, relv_vars):
        """returns the relevant messages from variables to a certain factor"""
        return [self.messagesVarToFactor[(var, fa)] for var in relv_vars]

    def getHamming(self):
        """ computes the hamming distance of the current codeword computed using the marginal MAP query"""
        codeWord = self.getMarginalMAP()
        hamm_dist = np.sum(codeWord)
        return hamm_dist

    def estimateMarginalProbability(self, var):
        '''
        Estimate the marginal probability of a single variable after running
        loopy belief propagation. (This method assumes runParallelLoopyBP has been run)

        param - var: a single variable index
        return: numpy array of size 2 containing the marginal probabilities 
                that the variable takes the values 0 and 1
        
        example: 
        >>> factor_graph.estimateMarginalProbability(0)
        >>> [0.2, 0.8]
        '''
        mult_ufunc = np.frompyfunc(self.mult_factors, 2, 1)
        relv_fa_locs = self.varToFactor[var]
        relv_fas = [self.factors[i] for i in relv_fa_locs]
        relv_msgs = [self.messagesVarToFactor[(var, r_fa)] for r_fa in relv_fas]
        return mult_ufunc.reduce(relv_msgs).normalize()

    def getOnesAssignmentProbs(self):
        """
        computes the probability to assign 1 at each of the variables
        :return:
        """
        Ps = np.empty((1, 2))
        for v in self.var:
            v_marg = self.estimateMarginalProbability(v).val
            Ps = np.vstack((Ps, v_marg))
        Ps = Ps[1:, :]
        return Ps[:, 1]

    def getMarginalMAP(self):
        '''
        In this method, the return value output should be the marginal MAP assignment for each variable.
        You may utilize the method estimateMarginalProbability.
        
        example: (N=2, 2*N=4)
        >>> factor_graph.getMarginalMAP()
        >>> [0, 1, 0, 0]
        '''
        Ps = np.empty((1, 2))
        for v in self.var:
            v_marg = self.estimateMarginalProbability(v)
            Ps = np.vstack((Ps, v_marg.val))
        Ps = Ps[1:, :]
        MAP = np.argmax(Ps, axis=1)
        return MAP

    def print(self):
        print('Variables:')
        for i in range(len(self.var)):
            print('  Variable {}: {}'.format(i, self.var[i]))
            print('     In factors:', self.varToFactor[i])
        print('Factors:')
        for i, f in enumerate(self.factors):
            print('  Factor {}: {}'.format(i, f))
            print('     vars=', self.factorToVar[i])
            print('     scope=', f.scope)
            print('     card=', f.card)
            print('     val=', f.val)
