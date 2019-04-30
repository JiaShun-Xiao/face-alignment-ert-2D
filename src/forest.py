from menpo.shape import PointCloud
from base import Regressor, RegressorBuilder

from  util import *
import numpy as np
    
class ForestBuilder(RegressorBuilder):
    def __init__(self,n_tree,tree_builder,get_pixel_value_builder):
        self.n_tree = n_tree
        self.tree_builder = tree_builder
        self.get_pixel_value_builder = get_pixel_value_builder
    
    def build(self, images, targets, current_shapes, mean_shape, logger):
        self.mean_shape = mean_shape
        # pixel coordinates were randomly selected in get_pixel_value
        get_pixel_value = self.get_pixel_value_builder.build()
        # the extracted pixel coordinate in mean shape
        pixel_mean_coords = self.mean_shape.points[get_pixel_value.lmark]+get_pixel_value.pixel_delta
        # Extract the pixel values from raw images
        pixel_vectors = np.array([get_pixel_value.get_feature(img, current_shape, 
 transform_to_mean_shape(current_shape,self.mean_shape).pseudoinverse()) for (img,current_shape) in zip(images,current_shapes)])
        trees = []
        logger.info("Building trees:")
        for i in range(self.n_tree):
            if i%100 == 0:
                logger.info("Finish {} trees".format(str(i)))
            tree = self.tree_builder.build(pixel_vectors,targets,pixel_mean_coords)
            # update the target
            targets -= [tree.apply(pixel_vector) for pixel_vector in pixel_vectors]
            trees.append(tree)
        return Forest(trees,mean_shape,get_pixel_value)

class Forest(Regressor):
    def __init__(self,trees,mean_shape,get_pixel_value):
        self.trees = trees
        self.mean_shape = mean_shape
        self.n_landmarks = mean_shape.n_points
        self.get_pixel_value = get_pixel_value
        
    def apply(self,image,current_shape):
        mean_to_shape = transform_to_mean_shape(current_shape,self.mean_shape).pseudoinverse()
        res = PointCloud(np.zeros((self.n_landmarks, 2)), copy=False)
        # Extract the pixel values from raw images
        pixel_vector = self.get_pixel_value.get_feature(image, current_shape, mean_to_shape)
        for tree in self.trees:
            res.points += tree.apply(pixel_vector).reshape((self.n_landmarks, 2))
        return mean_to_shape.apply(res)
