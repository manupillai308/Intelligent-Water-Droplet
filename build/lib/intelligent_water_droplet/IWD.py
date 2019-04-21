#!/usr/bin/env python
# coding: utf-8
import numpy as np
class Edge:
    """
    Class Edge:
        soil - soil content in edge of the graph 
        a - paramater a_s (default 1) 
        b - parameter b_s (default 0.01)
        c - parameter c_s (default 1)
    """
    def __init__(self, soil, a=1, b=0.01, c=1):
        self.soil = soil
        self.a = a
        self.b = b
        self.c = c

class Droplet:
    """
    Class Droplet:
        velocity - velocity of droplet
        a - parameter a_v (default 1)
        b - parameter b_v (default 0.01)
        c - parameter c_v (default 1)
    """
    def __init__(self, init_velocity, a=1, b=0.01, c=1):
        self.velocity = init_velocity
        self.a = a
        self.b = b
        self.c = c
        self.visited_nodes = []
        self.soil = 0


class Graph:
    """
    Class Graph:
        nodes - number of nodes in graph
        graph - an adjacency matrix of type numpy array of shape (nodes, nodes)
        init_soil - initial soil on each edges of the graph
    """
    def __init__(self, nodes, init_soil=10000):
        self.nodes = nodes
        self.graph = np.zeros((nodes, nodes))
        self.init_soil = init_soil
        self.edges = np.array([[Edge(self.init_soil) for _ in range(self.nodes)] for _ in range(self.nodes)])
    
    def build(self, fro, to, soil = 10000):
        self.graph[fro][to] = 1
        self.edges[fro][to].soil = soil
        return self



class IWD:
    """
    Class IWD:
        graph - a Graph object
        HUD - a function that takes to,fro as arguments and returns the local heuristics of the task.
            eg. HUD(to, fro)
        quality - a function that takes the list of node indexs of the path taken and returns its quality.
            The bigger the value, the better the quality of the path.
            eg. quality([0,1,2]) <- here the path is from node 0-1-2
        n_droplets - number of droplets to be used (defaults to number of nodes in graphs)
        rho_n - parameter rho_n (default 0.9)
        rho_iwd - parameter rho_iwd (default 0.9)
        
    """
    def __init__(self, graph, HUD, quality, n_droplets=None, rho_n=0.9, rho_iwd = 0.9):
        if n_droplets is None:
            import sys
            self.n_droplets = graph.nodes
            sys.stderr.write("Warning: n_droplets set to %d (number of nodes in graph)" % graph.nodes)
        else:
            self.n_droplets = n_droplets
        if not isinstance(graph, Graph):
            raise TypeError("graph must be of type Graph")
        self.HUD = HUD
        self.quality = quality
        self.graph = graph
        self.total_best = -np.inf
        self.iter_count = 1
        self.rho_n = rho_n
        self.rho_iwd = rho_iwd
        if self.rho_iwd > 1 or self.rho_iwd < 0:
            raise ValueError("rho_iwd should be between [0,1]")
    
    def init_graph(self, ni, alpha_i, total_best, tou):
        raise NotImplementedError("init_graph is not in standard IWD, try using IWD_TSP")
    
    def evaluate(self, init_velocity=200, ni = None, alpha_i=0.1, max_iter=None, thres_best=None):
        """
        evaluate:
            Arguments:
                init_velocity - The initial velocity of droplets (default 200)
                ni - Number of iteration after which the soil should be reinitialized in the graph (skip in case of standard IWD)
                alpha_i - Reinitializing parameter of soil in the graph (skip in case of standard IWD)
                max_iter - Maximum iterations to run
                thres_best - Threshold of quality after which the algorithm should stop.
            Returns:
                best_quality, best_path
        """
        if max_iter is None and thres_best is None:
            raise ValueError("max_iter and thres_best cannot be both None")
        if max_iter is not None:
            self.max_iter = max_iter
        if thres_best is not None:
            self.thres_best = thres_best
        else:
            self.thres_best = -np.inf
        total_best_q = -np.inf
        total_best = None
        tou = np.random.uniform(0,1)
        while self.iter_count <=  self.max_iter or total_best_q <= self.thres_best:
            if ni is not None and self.iter_count % ni == 0:
                self.init_graph(ni, alpha_i, total_best, tou)
            self.droplets = [Droplet(init_velocity) for _ in range(self.n_droplets)]
            iteration_q = -np.inf
            iteration_best = None
            iteration_soil = None
            for droplet_ix in range(len(self.droplets)):
                if start is None:
                    cur_node = np.random.randint(self.graph.nodes)
                else:
                    if not isinstance(start, int) or start >= self.graph.nodes:
                        raise ValueError("starting node must be an integer value and must be in range(0, %d)" % self.graph.nodes)
                    cur_node = start
                self.droplets[droplet_ix].visited_nodes.append(cur_node)
                while len(self.droplets[droplet_ix].visited_nodes) < self.graph.nodes:
                    probability = self.get_next_node_probability(cur_node, droplet_ix)
                    try:
                        next_node = np.random.choice(range(self.graph.nodes), p=probability)
                    except ValueError:
                        import sys
                        sys.stderr("isolated node %d, continuing with next droplet" % cur_node)
                        break
                    self.droplets[droplet_ix].visited_nodes.append(next_node)
                    self.droplets[droplet_ix].velocity+=self.droplets[droplet_ix].a/(self.droplets[droplet_ix].b + self.droplets[droplet_ix].c*np.square(self.graph.edges[cur_node][next_node].soil))
                    edge = self.graph.edges[cur_node][next_node]
                    del_soil = edge.a / (edge.b + edge.c * np.square(self.get_time(cur_node, next_node, droplet_ix)))
                    self.graph.edges[cur_node][next_node].soil = (1 - self.rho_n)*self.graph.edges[cur_node][next_node].soil - (self.rho_n * del_soil)
                    self.droplets[droplet_ix].soil+=del_soil
                    cur_node = next_node 
    
            for droplet in self.droplets:
                q = self.quality(droplet.visited_nodes)
                if iteration_q < q :
                    iteration_q = q
                    iteration_best = droplet.visited_nodes
                    iteration_soil = droplet.soil
            
            cur_node = iteration_best[0]
            for next_node in iteration_best[1:]:
                self.graph.edges[cur_node][next_node].soil = ((1 + self.rho_iwd) * self.graph.edges[cur_node][next_node].soil) - (self.rho_iwd * iteration_soil * 1/(len(iteration_best)-1))
                cur_node = next_node
            
            if total_best_q < iteration_q:
                total_best = iteration_best
                total_best_q = iteration_q
            
            self.iter_count+=1
                
        return total_best_q, total_best
    
    def get_time(self, i, j, droplet_ix):
        return self.HUD(i, j)/self.droplets[droplet_ix].velocity
    
    def get_next_node_probability(self, i, droplet_ix):
        probability = []
        s = 0
        for k in range(self.graph.nodes):
            if self.graph.graph[i][k] == 1 and k not in self.droplets[droplet_ix].visited_nodes:
                f = self.get_f(self.graph.edges[i][:], k, droplet_ix)
                s+=f
                probability.append(f)
            else:
                probability.append(0)
        return np.array(probability)/s if s!=0 else np.array(probability)
            
    def get_f(self, soil, j, droplet_ix, es=1e-10):
        return 1/(es+self.get_g(soil, j, droplet_ix))
    
    def get_g(self, soil, j, droplet_ix):
        min_soil = np.inf
        for i in range(self.graph.nodes):
            if i not in self.droplets[droplet_ix].visited_nodes:
                min_soil = min(min_soil, soil[i].soil)
        return soil[j].soil if min_soil>=0 else soil[j].soil - min_soil
        



class IWD_TSP(IWD):
    """
    Class IWD_TSP: The class is a variation of IWD algorithm optimized for TSP(Travelling Salesman Problem) like tasks.
        
        graph - a Graph object
        HUD - a function that takes to,fro as arguments and returns the local heuristics of the task.
            eg. HUD(to, fro)
        quality - a function that takes the list of node indexs of the path taken and returns its quality.
            The bigger the value, the better the quality of the path.
            eg. quality([0,1,2]) <- here the path is from node 0-1-2
        n_droplets - number of droplets to be used (defaults to number of nodes in graphs)
        rho_n - parameter rho_n (default 0.9)
        rho_iwd - parameter rho_iwd (default 0.9)
        
    """
    def __init__(self, graph, HUD, quality, n_droplets=None, rho_n=0.9, rho_iwd = 0.9):
        super().__init__(graph, HUD, quality, n_droplets, rho_n, rho_iwd)
    
    def evaluate(self, init_velocity=200, ni = None, alpha_i=0.1, max_iter=None, thres_best=None):
        """
        evaluate:
            Arguments:
                init_velocity - The initial velocity of droplets (default 200)
                ni - Number of iteration after which the soil should be reinitialized in the graph (skip in case of standard IWD)
                alpha_i - Reinitializing parameter of soil in the graph (skip in case of standard IWD) (default 0.1)
                max_iter - Maximum iterations to run
                thres_best - Threshold of quality after which the algorithm should stop.
            Returns:
                best_quality, best_path
        """
        if ni is None:
            raise ValueError("Ni (Number of iterations for soil reinitialisation) must be set for TSP-like problems")
        return super().evaluate(init_velocity, ni, alpha_i, max_iter, thres_best)
    
    def init_graph(self, ni, alpha_i, total_best, tou):
        arr = []
        i = total_best[0]
        for j in total_best[1:]:
            arr.append((i, j))
            i = j
        for i in range(self.graph.nodes):
            for j in range(self.graph.nodes):
                if self.graph.graph[i][j] == 1:
                    if (i,j) in arr:
                        self.graph.edges[i][j].soil = alpha_i*tou*self.graph.init_soil
                    else:
                        self.graph.edges[i][j].soil = self.graph.init_soil
                





