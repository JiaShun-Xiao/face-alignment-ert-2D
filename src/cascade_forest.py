from copy import deepcopy
from base import Regressor, RegressorBuilder
from util import *
from get_pixel_value import GetPixelValueBuild
from tree import TreeBuilder
from forest import ForestBuilder
import hickle

logger = configure_logging("cascade_forest_training")


class CascadeForestBuilder(RegressorBuilder):
    def __init__(self,n_landmarks,n_forests,n_trees,tree_depth,n_perturbations,n_test_split,n_pixels,kappa,lr):
        self.n_landmarks = n_landmarks
        self.n_forests = n_forests
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.n_perturbations = n_perturbations
        self.n_test_split = n_test_split
        self.n_pixels = n_pixels
        self.kappa = kappa
        self.lr = lr

    def build(self,images,gt_shapes,boxes):
        self.get_pixel_value_builder = GetPixelValueBuild(n_landmarks=self.n_landmarks,n_pixels=self.n_pixels,kappa=self.kappa)
        self.tree_builder = TreeBuilder(depth=self.tree_depth,n_test_split=self.n_test_split,lr=self.lr)
        self.forest_build = ForestBuilder(self.n_trees,self.tree_builder,self.get_pixel_value_builder)
        
        self.mean_shape = centered_mean_shape(gt_shapes)
        logger.info("Generating initial shapes")
        current_shapes = np.array([fit_shape_to_box(self.mean_shape, box) for box in boxes])
        logger.info("Perturbing initial estimates, perturbations times: {}".format(str(self.n_perturbations)))
        if self.n_perturbations > 1:
            images, current_shapes, gt_shapes, boxes = perturb_shapes(images, current_shapes, gt_shapes, boxes, 
                                                                      self.n_perturbations)
        
        logger.info("Size of augmented dataset after perturbations: {} images.".format(str(len(images))))
        
        cascade_forests = []
        for i in range(self.n_forests):
            shape_residual = [gt_shapes[i].points - current_shapes[i].points for i in range(len(images))]
            targets = np.array([transform_to_mean_shape(current_shapes[i],
                self.mean_shape).apply(shape_residual[i]).reshape((2*self.n_landmarks,)) for i in range(len(images))])
            forest = self.forest_build.build(images, targets, current_shapes, self.mean_shape, logger)
            # update current estimate of shapes
            for j in range(len(images)):
                current_shapes[j].points += forest.apply(images[j],current_shapes[j]).points
            cascade_forests.append(forest)
            logger.info("Built Forest {}".format(i))
            #model = CascadeForest(self.n_landmarks,cascade_forests,self.mean_shape)
            #hickle.dump(model, "../model/ert_ibug_train_{}_forests.hkl".format(i+1))
            
        return CascadeForest(self.n_landmarks,cascade_forests,self.mean_shape)
            
class CascadeForest(Regressor):
    def __init__(self,n_landmarks,cascade_forests,mean_shape):
        self.n_landmarks = n_landmarks
        self.cascade_forests = cascade_forests
        self.mean_shape = mean_shape
        
    def apply(self,image,boxes):
        initial_shapes = np.array([fit_shape_to_box(self.mean_shape, box) for box in boxes])
        shapes = deepcopy(initial_shapes)
        for i, shape in enumerate(shapes):
            for forest in self.cascade_forests:
                offset = forest.apply(image, shape)
                shape.points += offset.points
        return initial_shapes,shapes
