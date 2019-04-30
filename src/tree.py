import numpy as np
from base import Regressor, RegressorBuilder
    
# build tree in forest

class TreeBuilder(RegressorBuilder):
    def __init__(self,depth,n_test_split,lr):
        self.depth = depth
        self.n_test_split = n_test_split
        self.n_nodes = (1 << (depth-1)) - 1
        self.n_nodes_leafs = (1 << depth) - 1
        self.lr = lr # learning rate
        
    def build(self,pixel_vectors,targets,pixel_mean_coords):
        node_sums = np.zeros(self.n_nodes_leafs,dtype=object) # residual in each node
        node_cnts = np.zeros(self.n_nodes_leafs) # num of images in each node
        node_buckets = np.zeros(self.n_nodes_leafs,dtype=object) # dividing images into two parts in each node
        node_buckets[0] = (0, len(targets))
        best_splits = []
        permute = np.arange(0, len(targets), dtype=int) # permute order of pixel 
        
        for i in range(self.n_nodes):
            split,begin,midpoint,end,best_sums,permute = self.find_best_split(pixel_vectors,targets,self.n_test_split,permute,
                                          pixel_mean_coords,node_buckets[i],node_sums[i], node_cnts[i])
            best_splits.append(split)
            node_buckets[2*i+1] = (begin,midpoint)
            node_buckets[2*i+2] = (midpoint,end)
            (node_sums[2*i+1], node_sums[2*i+2]) = best_sums
            node_cnts[2*i+1] = (midpoint-begin)
            node_cnts[2*i+2] = (end-midpoint)
        
        leaves = np.zeros(shape=(1 << (self.depth-1), len(targets[0])))
        for i in range(self.n_nodes, (1 << self.depth) - 1):
            if node_cnts[i] != 0:
                leaves[i - self.n_nodes] = self.lr*node_sums[i] / node_cnts[i]
                
        return Tree(best_splits, leaves, self.depth)
    
    def gene_candidate_split(self,pixel_mean_coords):
        while True:
            u,v = np.random.randint(0, pixel_mean_coords.shape[0], 2)
            dist = np.absolute(np.linalg.norm(pixel_mean_coords[u]-pixel_mean_coords[v]))
            prob = np.exp(-dist/.1)
            if u != v and prob > np.random.random():
                break
        thresh = np.random.uniform(-0.25, 0.25)
        return int(u), int(v), thresh
    
    def find_best_split(self,pixel_vectors,targets,n_test_split,permute,pixel_mean_coords,node_bucket,node_sum,node_cnt):
        candidate_splits = np.array([self.gene_candidate_split(pixel_mean_coords) for _ in range(n_test_split)])
        pixel1 = np.array(candidate_splits[:, 0], dtype=int)
        pixel2 = np.array(candidate_splits[:, 1], dtype=int)
        begin,end = int(node_bucket[0]),int(node_bucket[1])
        divisions = (pixel_vectors[permute[begin:end]][:, pixel1] - 
                         pixel_vectors[permute[begin:end]][:, pixel2] > candidate_splits[:, 2]).transpose()
        
        best_score = -1
        best_division_index = 0
        best_midpoint = begin
        best_sums = (np.zeros(len(targets[0])), np.zeros(len(targets[0])))
        
        for i, division in enumerate(divisions):
            right_sum = targets[permute[begin:end]][division].sum(axis=0)
            right_cnt = float(np.count_nonzero(division))

            left_sum = node_sum - right_sum
            left_cnt = node_cnt - right_cnt
            lcnt = left_cnt
            rcnt = right_cnt
            if right_cnt == 0:
                rcnt = 1
            if left_cnt == 0:
                lcnt = 1
            score = left_sum.dot(left_sum)/lcnt + right_sum.dot(right_sum)/rcnt
            if score > best_score:
                best_division_index = i
                best_midpoint = begin + left_cnt
                best_score = score
                best_sums = (left_sum, right_sum)
        
        # peremute the index in current bucket
        ind = np.argsort(divisions[best_division_index])
        permute[begin:end] = permute[begin:end][ind] 

        return candidate_splits[best_division_index], int(begin), int(best_midpoint), int(end), best_sums, permute

    
class Tree(Regressor):
    def __init__(self, splits, leaves, depth):
        self.splits = splits
        self.leaves = leaves
        self.depth = depth
        
    def apply(self,pixel_vector):
        node = 0
        for k in range(self.depth-1):
            i, j, thresh = self.splits[node]
            i,j = int(i),int(j)
            node = 2*node+1
            if pixel_vector[i] - pixel_vector[j] > thresh:
                node += 1
        return self.leaves[node - len(self.splits)]
    
    