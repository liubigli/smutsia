import numpy as np
from numbers import Number

class Node:
    def __init__(self, node_id=-1, parent=-1, childs=None):
        self.node_id = node_id
        self.parent = parent
        self.childs = childs

    @property
    def node_id(self):
        return self.__node_id__

    @node_id.setter
    def node_id(self, node_id):
        self.__node_id__ = int(node_id)

    @property
    def parent(self):
        return self.__parent__

    @parent.setter
    def parent(self, parent):
        self.__parent__ = int(parent)

    @property
    def childs(self):
        return self.__childs__

    @childs.setter
    def childs(self, childs=None):
        if childs is None:
            self.__childs__ = list()
        else:
            childs = [int(c) for c in childs]
            self.__childs__ = childs

    def add_child(self, child):
        self.childs.append(int(child))

    def remove_child(self, child):
        self.childs.remove(int(child))

class HierarchyNode(Node):
    def __init__(self, node_id=-1, parent=-1, childs=None, root_id=-1, value=1, depth=-1, threshold=-1):
        Node.__init__(self, node_id, parent, childs)
        self.root_id = root_id
        self.value = value
        self.depth = depth
        self.threshold = threshold

    @property
    def root_id(self):
        return self.__root_id__

    @root_id.setter
    def root_id(self, root_id):
        self.__root_id__ = int(root_id)

    @property
    def value(self):
        return self.__value__

    @value.setter
    def value(self, value):
        self.__value__ = int(value)

    @property
    def depth(self):
        return self.__depth__

    @depth.setter
    def depth(self, depth):
        self.__depth__ = int(depth)

    @property
    def threshold(self):
        return self.__threshold__

    @threshold.setter
    def threshold(self, threshold):
        if isinstance(threshold, Number):
            self.__threshold__ = threshold
        else:
            raise TypeError("threshold parameter must be a number")

    def __str__(self):
        return "{ root_id: %s, parent: %s, childs: %s, value: %s, depth: %s, threshold: %s }" % (
        self.root_id, self.parent, self.childs, self.value, self.depth, self.threshold)

    def __repr__(self):
        return self.__str__()

class HierarchyTree:
    def __init__(self, nodes=None):
        self.nodes = nodes
        if nodes is None:
            self.nmb_nodes = 0
            self.max_id = -1
        else:
            self.nmb_nodes = len(nodes)
            # in this implementation of the HierarchyTree the root will always be the last node in node
            self.max_id = len(nodes) - 1

    @property
    def nodes(self):
        return self.__nodes__

    @nodes.setter
    def nodes(self, nodes=None):
        print(nodes)
        if nodes is None:
            self.__nodes__ = dict()
        else:
            if isinstance(nodes, dict):
                for k,v in nodes.items():
                    if not isinstance(k, Number):
                        raise TypeError("All the keys of dictionary nodes must be integers")
                    if not isinstance(v, HierarchyNode):
                        raise TypeError("All the values of dictionary nodes must be HierarchyNode")
                self.__nodes__ = nodes
            else:
                raise TypeError("Nodes must be a dictionary")


    @property
    def nmb_nodes(self):
        return self.__nmb_nodes__

    @nmb_nodes.setter
    def nmb_nodes(self, nmb_nodes):
        self.__nmb_nodes__ = int(nmb_nodes)

    def add_node(self, **kwargs):
        """
        Method that add a node to the hierarchical hierarchy
        :param kwargs:
        :return:
        """
        node = kwargs.get('node', None)
        if isinstance(node, HierarchyNode):
            self.nodes[self.nmb_nodes] = node
            self.nmb_nodes += 1
        else:
            node_id = kwargs.get('node_id', -1)
            node_id = node_id if node_id >= 0 else self.nmb_nodes
            root_id = kwargs.get('root_id', -1)
            parent = kwargs.get('parent', -1)
            childs = kwargs.get('childs', None)
            value = kwargs.get('value', 1)
            depth = kwargs.get('depth', -1)
            threshold = kwargs.get('threshold', -1)


            self.nodes[self.max_id + 1] = HierarchyNode(node_id=node_id,
                                                       parent=parent,
                                                       childs=childs,
                                                       root_id=root_id,
                                                       value=value,
                                                       depth=depth,
                                                       threshold=threshold)


            self.max_id +=1
            self.nmb_nodes += 1

    def delete_node(self, node_id):
        """
        Methods that deletes nodes from id
        :param node_id: id of the node to delete
        :return:
        """
        if node_id == self.max_id:
            print("Warning!! We are deleting the root of the hierarchy")

        del self.nodes[node_id]
        # updating the number of nodes in the hierarchy
        self.nmb_nodes -= 1

    def remap_hierarchical_tree(self):
        """
        Method that remap all the ids of the nodes in the hierarchy from 1 to n
        :return:
        """

        # before that we remap we find parents childs and depths for all the nodes in the trees
        # todo: add a check in order to avoid the recomputing of depths and parents
        self.__initialize_parents_and_childs()

        if self.max_id == self.nmb_nodes - 1:
            # in this case we don't need to remap the hierarchy because all the node are well numbered
            return




        remap_table = -1 * np.ones(self.max_id + 1, dtype=np.int32)

        i = 0  # incrementing variable
        for k in self.nodes:
            remap_table[k] = i
            i += 1

        remapped_tree = {}

        for k in self.nodes:
            remapped_tree[remap_table[k]] = HierarchyNode(root_id=remap_table[self.nodes[k].root_id],
                                                          parent=remap_table[self.nodes[k].parent],
                                                          childs=remap_table[self.nodes[k].childs].tolist(),
                                                          value=self.nodes[k].value,
                                                          depth=self.nodes[k].depth,
                                                          threshold=self.nodes[k].threshold)

        # uptading all the hierarchy's parameters
        self.nodes = remapped_tree
        self.nmb_nodes = len(self.nodes)
        self.max_id = len(self.nodes) - 1



    def __initialize_parents_and_childs(self):
        """
        Auxiliary method used to initialize all the values of depth, parents id and childs for all the nodes of the hierarchy
        :return:
        """
        if self.max_id == -1:
            if self.nmb_nodes == 0:
                return
            else:
                raise IndexError("The current hierarchy has no valid root assigned")

        root = self.nodes[self.max_id]
        root.depth = 0
        root.parent = -1

        # index to traverse the hierarchy
        i = 0
        j = 1
        # list of euler trip
        traverse_list = -np.ones(self.nmb_nodes, dtype=np.uint64)
        traverse_list[i] = self.nmb_nodes - 1
        # BFS
        while i < self.nmb_nodes:
            node_id = traverse_list[i]
            if len(self.nodes[node_id].childs) > 0:
                for child in self.nodes[node_id].childs:
                    traverse_list[j] = child
                    j += 1
                    self.nodes[child].depth = self.nodes[node_id].depth + 1
                    self.nodes[child].parent = node_id
            i += 1

    def get_lca(self, n, m):
        """
        Basic function that compute the Least Common Ancestor between two nodes in the Hierarchical hierarchy
        :param n: id of the first node
        :param m: id of the second node
        :return: id of the LCA
        """

        while self.nodes[n].depth > self.nodes[m].depth:
            # going up to the branch of node n while it is deeper that m
            n = self.nodes[n].parent

        while self.nodes[m].depth > self.nodes[m].depth:
            # going up to the branch of node m while it is deeper that n
            m = self.nodes[m].parent

        # case in which one of the two node was the parent of the other
        if n == m:
            return n

        while self.nodes[n].parent != self.nodes[m].parent:
            n = self.nodes[n].parent
            m = self.nodes[m].parent

        return n

    def __str__(self):
        return self.nodes.__str__()