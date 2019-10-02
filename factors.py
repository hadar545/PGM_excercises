###############################################################################
# Utility functions for manipulating factors; ported from Daphne Koller's
# Matlab utility code
# author: Ya Le, Billy Jun, Xiaocheng Li
# date: Jan 25, 2018
# You don't need to modify this file for PA2
# April 16, 2019: Major refactoring and optimization by Eitan Richardson
###############################################################################
import numpy as np


def intersection_indices(a, b):
    """
    :param list a, b: two lists of variables from different factors.

    returns a tuple of
        (indices in a of the variables that are in both a and b,
        indices of those same variables within the list b)
        For example, intersection_indices([1,2,5,4,6],[3,5,1,2]) returns
        ([0, 1, 2], [2, 3, 1]).
    """
    mapA = [i for i, v in enumerate(a) if v in b]
    mapB = [b.index(a[i]) for i in mapA]
    return mapA, mapB


class Factor:
    def __init__(self, f=None, scope=[], card=[], val=None, name="[unnamed]"):
        """
        :param Factor f: if this parameter is not None, then the constructor makes a
            copy of it.
        :param list scope: a list of variable names that are in the scope of this factor
        :param list card: a list of integers coresponding to the cardinality of each variable
            in scope
        :param np.ndarray val: an array coresponding to the values of different assignments
            to the factor. val is a numpy.ndarray of shape self.card. Therefore, if this factor is over
            three binary variables, self.val will be an array of shape (2,2,2)
        :param str name: the name of the factor.  Useful for debugging only--no functional
            purpose.
        """
        assert len(scope) == len(card)
        assert val is None or list(val.shape) == card

        # self.scope: a list of the variables over which this Factor defines a distribution
        self.scope = scope

        # self.card: the cardinality of each variable in self.scope
        self.card = card

        # use the name field for debugging, imo
        self.name = name

        self.val = val

        if f is not None:
            self.scope = list(f.scope)
            self.card = list(f.card)
            self.val = np.array(f.val, copy=True)
            self.name = f.name

    def compose_factors(self, f, operator, opname="op"):
        """
        Returns a factor that is the result of composing this
        factor under the operator specified by the parameter operator.
        This is a general function that can be used to sum/multiply/etc factors.

        :param Factor f: the factor by which to multiply/sum/etc this factor.
        :param function operator: a function taking two arrays and returning a third.
        :param str opname: a string naming the operation.  Optional but nice for visualization.

        :rtype: Factor

        --------------------------------------------------------------------------------
        You may find the following functions useful for this implementation:
            -intersection_indices
        Depending on your implementation, the numpy function np.reshape and the numpy.ndarray
        field arr.flat may be useful for this as well, when dealing with the duality between
        the two representations of the values of a factor.
        """

        g = Factor() # modify this to be the composition of two Factors and then return it
        #g.name = "(%s %s %s)" % (self.name, opname, f.name)
        g.name = 'x'

        if len(f.scope) == 0:
            return Factor(self)
        if len(self.scope) == 0:
            return Factor(f)
        g.scope = list(set(self.scope) | set(f.scope))

        # Below regamarole just sets the cardinality of the variables in the scope of g.
        g.card = np.zeros(len(g.scope), dtype='int32')
        _, m1 = intersection_indices(self.scope, g.scope)
        g.card[m1] = self.card
        _, m2 = intersection_indices(f.scope, g.scope)
        g.card[m2] = f.card

        # Perform the actual operation
        g.val = operator(self.get_reshaped_val(g.scope), f.get_reshaped_val(g.scope))
        return g

    def get_reshaped_val(self, scope):
        '''
        Return an expanded val ndarray with matching dimensions to scope and includes singeton (1) dimensions for
        all variables not in current factor.
        e.g. if self.scope is [3, 1], scope = [1, 2, 3] and all variables are binary, the function will perform:
             self.val.transpose([1, 0]).reshape([2, 1, 2])
        :param scope: The full (expanded) list of variables
        :return: A new shape
        '''
        transpose_op = [list(self.scope).index(v) for v in scope if v in self.scope]
        new_shape = [self.card[list(self.scope).index(v)] if v in self.scope else 1 for v in scope]
        return self.val.transpose(transpose_op).reshape(new_shape)

    def sum(self, f):
        """
        Returns a factor that is the result of adding this factor with factor f.

        :param Factor f: the factor by which to multiply this factor.
        :rtype: Factor
        """
        return self.compose_factors(f, operator=lambda x, y: x+y, opname = "+").normalize()

    def multiply(self, f):
        """
        Returns a factor that is the result of multiplying this factor with factor f.

        Looking at Factor.sum() might be helpful to implement this function.  This is
        very simple, but I want to make sure you know how to use lambda functions.

        :param Factor f: the factor by which to multiply this factor.
        :rtype: Factor
        """
        return self.compose_factors(f, operator=lambda x, y: x*y, opname = "*").normalize()

    def divide(self, f):
        """
        Returns a factor that is the result of dividing this factor with factor f.

        Looking at Factor.sum() might be helpful to implement this function.  This is
        very simple, but I want to make sure you know how to use lambda functions.

        :param Factor f: the factor by which to divide this factor.
        :rtype: Factor
        """
        return self.compose_factors(f, operator=lambda x, y: x/y, opname = "/").normalize()

    def marginalize_all_but(self, var):
        """
        returns a copy of this factor marginalized over all variables except those
        present in var

        Inputs:
        - var (set of ints): indices of variables that will NOT be marginalized over

        Outputs:
        - g (Factor): the new factor marginalized over all variables except those
            present in var

        """
        if len(self.scope) == 0 or len(var) == 0 or list(self.scope) == list(var):
            return Factor(self)
        for v in var:
            if v not in self.scope:
                return Factor()
        g = Factor()
        g.scope = list(var)
        g.card = np.zeros(len(g.scope), dtype='int32')
        ms, mg = intersection_indices(self.scope, g.scope)
        for i, msi in enumerate(ms):
            g.card[mg[i]] = self.card[msi]

        # Eitan's alternative implementation...
        marginalization_dims = [i for i, v in enumerate(self.scope) if v not in var]
        common_vars = [v for v in self.scope if v in var]
        transpose_op = [common_vars.index(v) for v in var]
        g.val = np.sum(self.val, axis=tuple(marginalization_dims)).transpose(transpose_op)
        return g

    def observe(self, var, val):
        """
        Returns a version of this factor with variable var observed as having taken on value val.
        if var is not in the scope of this Factor, a duplicate of this factor is returned.

        :param str var: the observed variable
        :param int val: the value that variable took on
        :return: a Factor corresponding to this factor with var observed at val

        This will involve zeroing out certain rows/columns, and may involve reordering axes.
        """
        f = Factor(self) #make copy.  You'll modify this.
        f.name = "(%s with variable %s observed as %s)"%(self.name, var, val)
        if var not in self.scope:
            return Factor(self)

        idx = f.scope.index(var)

        order = range(len(f.scope))

        varLoc = f.scope.index(var)
        order[0] = varLoc
        order[varLoc] = 0
        factor = f.val
        permuted = np.transpose(factor, order)
        for j in range(f.card[idx]):
            if j != val:
                permuted[j].fill(0.0)
        return f

    def normalize(self):
       """
       Normalize f to a probability distribution
       """
       f = Factor(self)
       f.val /= np.sum(f.val.flatten())
       return f

    def __repr__(self):
        """
        returns a descriptive string representing this factor!
        """
        r = "Factor object with scope %s and corresponding cardinalities %s"%(self.scope, self.card)
        r += "\nCPD:\n" + str(self.val)
        if self.name:
            r = "Factor %s:\n"%self.name + r
        return r + "\n"

    def __str__(self):
        """
        returns a nice string representing this factor!  Note that we can now use string formatting
        with %s and this will cast our class into somethign nice and readable.
        """
        return self.name   

