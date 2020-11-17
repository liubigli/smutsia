from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.sparse import csr_matrix, find


class DasguptaCost:
    def __init__(self, tree, graph):
        # the variable hierarchy represents the hierarchy
        self.tree = tree

        # the variable graph is represented as a sparse matrix
        self.graph = graph

        # Euler trip of the hierarchy; depth of the nodes; vector of representant
        self.__E__, self.__L__, self.__R__ = self.__reduce_lca_to_rmq__()

        # length of the euler path
        self.__length_arrays__ = len(self.__E__)

        # blocks size
        self.__block_size__ = np.ceil(np.log(self.__length_arrays__) / 2).astype(int)

        # normalized sequence table
        self.__rmq_normalized_table__ = self.__generate_rmq_normalized_table__(self.__block_size__)


        # vector of partitions
        self.__table__ = self.__partition_vector__(self.__L__)

    @property
    def tree(self):
        return self.__tree__

    @tree.setter
    def tree(self, tree):
        self.__tree__ = tree

    @property
    def graph(self):
        return self.__graph__

    @graph.setter
    def graph(self, graph):
        if isinstance(graph, csr_matrix):
            self.__graph__ = graph
        else:
            raise TypeError("Graph must be represented as a csr matrix")

    def __reduce_lca_to_rmq__(self):
        """Function that takes a hierarchy and the problem of finding a lca between two node to rmq (range minimum query)
           in an array
        """
        # number of nodes in the hierarchy
        n = len(self.tree)
        # we are going to traverse the hierarchy in with a DFS and keeping track of the euler tour
        euler_trip = np.zeros(2 * n - 1, dtype=np.int32)
        # vector of labels
        labels = np.zeros(2 * n - 1, dtype=np.int32)
        # vector of representatives
        representatives = np.zeros(n, dtype=np.int32)

        # label of the hierarchy's root
        root = max(self.tree)

        # initializing
        i = 0
        j = 0  # index for the representative vector

        # vector of nodes to visit. We will keep track of the nodes to visit at step i
        visit_list = -np.ones(2 * n - 1)
        # in the first iteration we will visit the root
        visit_list[i] = root
        # set to 0 the depth of the root node
        self.tree[root]['depth'] = 0
        self.tree[root]['parent'] = -1

        # DFS
        while i < 2 * n - 2:
            node = visit_list[i].astype(int)
            euler_trip[i] = node
            labels[i] = self.tree[node]['depth']

            if self.tree[node].get('visited', None) is None:
                # setting representative value
                representatives[node] = i
                self.tree[node]['visited'] = 1
                # updating the counter
                j += 1

            if len(self.tree[node]['childs']) > 0:
                # if the node has at least one child we
                child = self.tree[node]['childs'].pop(0)
                # setting depth and parent of child node
                self.tree[child]['depth'] = self.tree[node]['depth'] + 1
                self.tree[child]['parent'] = node.astype(int)
                visit_list[i + 1] = child
            else:
                visit_list[i + 1] = self.tree[node]['parent']

            i += 1

        # the final node in the euler tour is always the root node
        euler_trip[2 * n - 2] = root
        labels[2 * n - 2] = 0

        return euler_trip, labels, representatives

    def __partition_vector__(self, A):

        # dimension of the second array
        m = np.ceil(self.__length_arrays__ / self.__block_size__).astype(int)

        # A1 contains the minimum in the ith block of A
        A1 = np.zeros(m, dtype=np.int64)

        # keeps track of where the minima in A1 come from
        B = np.zeros(m, dtype=np.int64)
        j = 0
        for i in range(0, self.__length_arrays__, self.__block_size__):
            if i >= (m-1)*self.__block_size__:
                # stupid way to solve the last part of the array
                # todo: solve here
                A1[j] = np.min(A[i:])
                B[j] = np.argmin(A[i:])
            else:
                # from the sequence we can reconstruct the indices of the normalized seq in the normalized table
                indx = self.__remap_to_table_index(A[i:i + self.__block_size__])
                # we store the minimum of the ith block
                A1[j] = self.__rmq_normalized_table__[0, self.__block_size__ - 1, 0, indx] + A[i]
                # B[j] is a position in the ith block in which value A1[i] occurs
                B[j] = self.__rmq_normalized_table__[0, self.__block_size__ - 1, 1, indx]

            j+=1

        return self.__sparse_table__(A1, B)

    def __remap_to_table_index(self, seq):
        """
        Method that convert a sequence into its binary representation and then to the corresponding base10 number
        :param seq: sequence of depths during the
        :return:
        """
        s = seq - seq[0]

        bit_list = [str(int((s[i] - s[i - 1]) * (s[i] - s[i - 1] + 1) / 2)) for i in range(1, len(seq))]

        return int(''.join(bit_list), 2)

    def __solve_rmq_for_normalized_sequence(self, normalized_sequence, table):
        """
        Method that solve RMQ for all the possible couples in a normalized sequence of length m

        :param normalized_sequence:
        :return:
        """
        # TODO: implement a more efficient method
        n = len(normalized_sequence)

        for i in range(n):
            for j in range(i, n):
                # in the first slice we store values while in the second slice we store the indices
                table[i, j, 0] = np.min(normalized_sequence[i:j+1]) # values
                table[i, j, 1] = np.argmin(normalized_sequence[i:j+1]) # indices

    def __generate_rmq_normalized_table__(self, m):
        # function that solves the +/- 1-RMQ problem for all the possible arrays of length m that starts with 0
        n_cases = 2 ** (m - 1)

        rmq_normalized_table = np.zeros((m, m, 2, n_cases))

        for n in range(n_cases):
            # all the possible +/-1-RMQ sequence are all the binary sequence of length m-1
            case = "{0:{fill}{length}b}".format(n, fill=0, length=m - 1)
            normalized_sequence = np.array([0] + [1 if y == '1' else -1 for y in case]).cumsum()
            self.__solve_rmq_for_normalized_sequence(normalized_sequence, table=rmq_normalized_table[:, :, :, n])

        return rmq_normalized_table

    def __sparse_table__(self, A, B):
        # length of array A
        n = len(A)
        # ln of length of A
        m = np.floor(np.log2(n)).astype(int)

        # sparse_table = np.zeros((n, m + 1, 3), dtype=np.int64)
        sparse_table = -np.ones((n, m + 1, 3), dtype=np.int64)
        sparse_table[:, 0, 0] = A
        sparse_table[:, 0, 1] = np.arange(n)
        sparse_table[:, 0, 2] = B
        for j in range(1,m+1):
            for i in range(0, n - 2 ** j + 1):
                if A[sparse_table[i, j - 1, 1]] <= A[sparse_table[i + 2 ** (j - 1), j - 1, 1]]:
                    # in the first slice we store values while in the second slice we store the indices
                    sparse_table[i, j, :] = sparse_table[i, j - 1, :]
                    # sparse_table[i, j, 1] = sparse_table[i, j - 1, 1]
                    # sparse_table[i, j, 2] = sparse_table[i, j - 1, 2]
                else:
                    # in the first slice we store indices while in the second slice we store the indices
                    sparse_table[i, j, :] = sparse_table[i + 2 ** (j - 1), j - 1, :]
                    # sparse_table[i, j, 1] = sparse_table[i + 2 ** (j - 1), j - 1, 1]
                    # sparse_table[i, j, 2] = sparse_table[i + 2 ** (j - 1), j - 1, 2]

        return sparse_table

    def query_sparse_table(self,i,j):
        """
        Function that return the minimum
        :param i:
        :param j:
        :return:
        """
        if i > j:
            return [np.inf, np.inf, np.inf]
        elif i ==j:
            return self.__table__[i, 0, :].astype(int)

        k = np.floor(np.log2(j-i)).astype(int)

        if self.__table__[i, k, 0] < self.__table__[j - 2 ** k + 1, k, 0]:
            # in the first coordinate the value while in the second the index
            return self.__table__[i, k, :]
        else:
            # as above in first position the value in the second the index
            return self.__table__[j - 2** k + 1, k, :].astype(int)

    def query_normalized_table(self, i, j, indx):
        """
        Function that return the minimum of interval i-j in the ith block
        :param block:
        :param i:
        :param j:
        :return:
        """
        return self.__rmq_normalized_table__[i, j, :, indx].astype(int)

    def getLCA(self, node1, node2):
        """
        Method to compute the LCA between two nodes in the hierarchy
        :param node1:
        :param node2:
        :return:
        """
        i = min(self.__R__[node1], self.__R__[node2])
        j = max(self.__R__[node1], self.__R__[node2])
        # get the index of the ith block
        block_i = np.floor(i / self.__block_size__).astype(int)
        # getting the index of the jth block
        block_j = np.floor(j / self.__block_size__).astype(int)

        # retrieving the ith seq
        seq_i = self.__L__[block_i*self.__block_size__:(block_i+1)*self.__block_size__]
        # retrieving the jth seq
        seq_j = self.__L__[block_j*self.__block_size__:(block_j+1)*self.__block_size__]
        # getting the index of the ith seq in the normalized table
        indx_i = self.__remap_to_table_index(seq_i)
        # getting the index of the jth sequ int the normalized table
        indx_j = self.__remap_to_table_index(seq_j)


        m1 = self.query_normalized_table(i%self.__block_size__, self.__block_size__ - 1, indx_i)
        m2 = self.query_normalized_table(0, j%self.__block_size__, indx_j)
        m3 = self.query_sparse_table(block_i + 1, block_j - 1)

        if m1[0] + seq_i[0] <= m2[0] + seq_j[0] and m1[0] + seq_i[0] <= m3[0]:
                # m1 is the minimum
                return self.__E__[block_i*self.__block_size__ + m1[1]]
        elif m2[0] + seq_j[0] <= m3[0]:
                # m2 is the minimum
                return self.__E__[block_j*self.__block_size__ + m2[1]]

        # if here is because m3 is the minimum
        return self.__E__[self.__block_size__*m3[1] + m3[2]]

    def compute_cost(self):
        """
        Method that compute the cost of the hierarchy according to the functional defined by
        Sanjoy Dasgupta in "A cost for similarity-based hierarchical clustering"
        <a href="https://cseweb.ucsd.edu/~dasgupta/papers/hier-cost.pdf"> link </a>
        :return:
        """
        [src, dst, weights] = find(self.graph)

        cost = 0

        for i,j,k in zip(src, dst, weights):
            lca = self.getLCA(i,j)
            val = self.tree[lca]['value']
            cost += k * self.tree[lca]['value']

        return cost
