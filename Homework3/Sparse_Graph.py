import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
import random

DEBUG = False

class SparseGraph:
    """
    SparseGraph is a class used to exploit the library scipy.sparse and works as
    a wrapper for all the methods applied on the graph.
    """
    def __init__(self, adjacency_matrix):
        self.Graph = False
        self.adjacency_matrix = adjacency_matrix
        self.num_nodes = np.size(adjacency_matrix,axis=0)
        self.nodes_list = np.arange(self.num_nodes)
        self.nodes_states = {}
        for node in range(self.num_nodes):
            self.nodes_states[node] = 'S'

    def set_node_state(self, node, state):
        self.nodes_states[node] = state

    def get_node_state(self, node):
        return self.nodes_states[node]

    def neighbors(self, node):
        """
        this method finds the neighbours of node exploiting the computational advantages of sparse matrices
        """
        return self.adjacency_matrix.indices[self.adjacency_matrix.indptr[node]:self.adjacency_matrix.indptr[node + 1]]

    def get_dictionaries(self):
        """
        returns a dictionary that couples 'state' -> list of nodes in that state
        """
        nodes_dictionary = {'S':[],'I':[],'R':[],'V':[],'NOVAX':[]}
        num_nodes_dictionary = {'S':0,'I':0,'R':0,'V':0,'NOVAX':0}
        for node, state in self.nodes_states.items():
            nodes_dictionary[state].append(node)
            num_nodes_dictionary[state] += 1
            if state != 'V':
                nodes_dictionary['NOVAX'].append(node)
                num_nodes_dictionary['NOVAX'] += 1
        return nodes_dictionary, num_nodes_dictionary
    
    def plot_Graph(self):
        if self.Graph == False:
            self.Graph = nx.from_numpy_array(self.adjacency_matrix.todense())
        color = {'S': 'blue', 'I': 'red', 'R': 'green', 'V': 'purple'}
        node_color = [color[self.nodes_states[i]] for i in self.nodes_list]
        nx.draw(self.Graph, with_labels=False, pos=nx.spring_layout(self.Graph), font_weight='bold', node_color=node_color, font_color='black', edge_color='gray', node_size=40)
        plt.show()


def random_graph(n, k):
    mat = np.zeros((n,n))

    # start with complete graph of k+1 nodes
    mat[0:k+1, 0:k+1] = np.ones((k+1,k+1)) - np.eye(k+1)

    # convert to sparse matrix
    adjacency_matrix = sp.lil_matrix(mat)

    # complete graph has all degrees k
    degrees = np.zeros(n)
    degrees[:k+1] = k    
    deg_tot = k * (k+1)
    
    # foreach new node, till ntotal nodes
    for node in range(k+1, n):

        # probabilities[i] is the probablity of attaching to node i
        probabilities = degrees[:node+1] / deg_tot

        # if k is even, num_links is k/2, 
        # if k is odd, for example 5, then num_links is 2 for even index nodes, 3 for odd index nodes
        num_links = int(k/2 + (node%2)*(k%2))

        # choose nodes to attach
        chosen_nodes = np.random.choice(np.arange(node+1), size=num_links, p=probabilities, replace=False)

        # the graph is undirected so the number of edges added is 2 times num_links 
        deg_tot += num_links*2
        degrees[chosen_nodes] += 1
        degrees[node] += num_links

        # Add the edges between the new node and the chosen nodes
        for node_link in chosen_nodes:
            # edge (i,j)
            adjacency_matrix[node, node_link] = 1
            # edge (j,i)
            adjacency_matrix[node_link, node] = 1
    return adjacency_matrix.tocsr()

def k_regular_graph(n, k):
    #sparse matrix
    matrix = sp.lil_matrix((n, n), dtype=int)

    # for each node, connect with the k/2 nearest nodes
    for i in range(n):
        for j in range(1, k // 2 + 1):
            neighbor1 = (i + j) % n
            neighbor2 = (i - j) % n

            # simmetric edges
            matrix[i, neighbor1] = 1
            matrix[neighbor1, i] = 1

            matrix[i, neighbor2] = 1
            matrix[neighbor2, i] = 1

    return matrix.tocsr()
        
