import util
import menpo.io as mio
import menpodetect
import numpy as np
import hickle
from src.cascade_forest import CascadeForestBuilder

# read images
ibugin = util.read_images("../../ibug/300W/train/", normalise=True)

face_detector = menpodetect.load_dlib_frontal_face_detector()

train_gt_shapes = util.get_gt_shapes(ibugin)
train_boxes = util.get_bounding_boxes(ibugin, train_gt_shapes, face_detector)

# n_landmarks: number of landmarskd
# n_forests: number of regressor in cascade
# n_trees: number of trees in each regressor
# tree_depth: tree depth
# n_perturbations: number of initializations for each training example
# n_test_split: number of randomly generated candidate split for each node of tree
# n_pixels: number of pixel locations are sampled from the image
# kappa: range of extracted pixel around the current estimated landmarks position
# lr: learning rate
cascade_forest_builder = CascadeForestBuilder(n_landmarks=68,n_forests=10,n_trees=500,
                                tree_depth=5,n_perturbations=20,n_test_split=20,n_pixels=400,kappa=.3,lr=.1)


# training model
model = cascade_forest_builder.build(ibugin, train_gt_shapes, train_boxes)
# save model
hickle.dump(model, "./model/ert_ibug_training.hkl")

# test model
ibug_exam = util.read_images("./ibug_test/image_060_1.*",normalise=True)
ibug_exam_shapes = util.get_gt_shapes(ibug_exam)
ibug_exam_boxes = util.get_bounding_boxes(ibug_exam, ibug_exam_shapes, face_detector)
ibug_exam = ibug_exam[0]
model = hickle.load("../model/ert_ibug_training.hkl")
init_shapes, fin_shapes = model.apply(ibug_exam,[ibug_exam_boxes[0]])
try:
    ibug_exam.landmarks.pop('dlib_0')
except:
    pass
ibug_exam_gt = deepcopy(ibug_exam)
ibug_exam_gt.view_landmarks()
