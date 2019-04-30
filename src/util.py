import menpo
from menpo import io as mio
from menpo.shape import PointCloud
from menpo.visualize import print_dynamic
import numpy as np
from menpo.transform import AlignmentSimilarity
import datetime
import os
import sys
import logging

def configure_logging(logger_name):
    LOG_LEVEL = logging.DEBUG
    log_filename = logger_name+'.log'
    importer_logger = logging.getLogger('importer_logger')
    importer_logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')

    fh = logging.FileHandler(filename=log_filename)
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(formatter)
    importer_logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(LOG_LEVEL)
    sh.setFormatter(formatter)
    importer_logger.addHandler(sh)
    return importer_logger

def computer_error(shape1,gt_shape):
    # distance between eyes
    eye_dis = np.linalg.norm(gt_shape.points[36]-gt_shape.points[45])
    return np.linalg.norm(shape1.points-gt_shape.points, axis=1).mean()/eye_dis

def transform_to_mean_shape(src, mean_shape):
    centered = PointCloud(src.points - src.centre(), copy=False)

    return AlignmentSimilarity(centered, mean_shape)

def normalize(ret):
    length = np.linalg.norm(ret)
    if length > 0:
        return ret / length
    return ret

def rand_unit_vector(dim):
    ret = np.random.randn(dim)
    return normalize(ret)

def get_gt_shapes(images):
    return np.array([image.landmarks['PTS'].lms for image in images])

def center_shape(shape):
    return PointCloud(2 * (shape.points - shape.centre()) / shape.range())

def centered_mean_shape(target_shapes):
    mean_shape = menpo.shape.mean_pointcloud(target_shapes)
    return center_shape(mean_shape)

def fit_shape_to_box(normal_shape, box):
    x, y = box.points[0]
    w, h = box.range()

    center_x = x + w/2.0
    center_y = y + h/2.0

    shape = normal_shape.points - normal_shape.centre()
    shape *= [0.9*h/2.0, 0.9*w/2.0]
    shape += [center_x, center_y]

    return PointCloud(shape)

def perturb_shapes(images, shapes, gt_shapes, boxes, n_perturbations):
    boxes = boxes.repeat(n_perturbations, axis=0)
    images = images.repeat(n_perturbations, axis=0)
    shapes = shapes.repeat(n_perturbations, axis=0)

    gt_shapes = gt_shapes.repeat(n_perturbations, axis=0)

    dx = np.random.uniform(low=-0.15, high=0.15, size=(len(shapes)))
    dy = np.random.uniform(low=-0.15, high=0.15, size=(len(shapes)))
    scale = np.random.normal(1, 0.07, size=(len(shapes)))
    normalized_offsets = np.dstack((dy, dx))[0]

    ret = []
    for i in range(len(shapes)):
        midpt = (gt_shapes[i].centre() + shapes[i].centre()) / 2
        ret.append(PointCloud((shapes[i].points-shapes[i].centre())*scale[i]
                              + midpt + shapes[i].range() * normalized_offsets[i]))

    return images, ret, gt_shapes, boxes

def perturb_init_shape(init_shape, num):
    ret = [init_shape]
    if num <= 1:
        return ret
    dx = np.random.uniform(low=-0.10, high=0.10, size=(num-1))
    dy = np.random.uniform(low=-0.10, high=0.10, size=(num-1))
    scale = np.random.normal(1, 0.07, size=(num-1))
    normalized_offsets = np.dstack((dy, dx))[0]

    for i in range(num-1):
        ret.append(PointCloud((init_shape.points-init_shape.centre())*scale[i] +
                              init_shape.centre() + init_shape.range() * normalized_offsets[i]))
    return ret

def is_point_within(pt, bounds):
    x, y = pt
    return bounds[0][0] <= x <= bounds[1][0] and bounds[0][1] <= y <= bounds[1][1]

def get_bounding_boxes(images, gt_shapes, face_detector):
    ret = []
    for i, (img, gt_shape) in enumerate(zip(images, gt_shapes)):
        #print_dynamic("Detecting face {}/{}".format(i, len(images)))
        boxes = face_detector(img)
        found = False
        for box in boxes:
            if is_point_within(gt_shape.centre(), box.bounds()):
                ret.append(box)
                found = True
                break
        if not found:
            ret.append(gt_shape.bounding_box())


    return np.array(ret)

#MAX_FACE_WIDTH = 500.0
def read_images(img_glob, normalise):
    # Read the training set into memory.
    images = []
    for img_orig in mio.import_images(img_glob, verbose=True, normalise=normalise):
        if not img_orig.has_landmarks:
            continue
        # Convert to greyscale and crop to landmarks.
        img = img_orig.as_greyscale(mode='average').crop_to_landmarks_proportion(0.5)
        #img = img.resize((MAX_FACE_WIDTH, img.shape[1]*(MAX_FACE_WIDTH/img.shape[0])))
        images.append(img)
    return np.array(images)

def get_median_shape(shapes):
    if len(shapes) == 1:
        return shapes[0]
    n_landmarks = len(shapes[0].points)
    ret = np.zeros(n_landmarks*2)

    if len(shapes) > 1:
        pts = np.array([s.as_vector() for s in shapes]).transpose()
        for k in range(n_landmarks*2):
            ret[k] = np.median(pts[k])
    return PointCloud(ret.reshape(n_landmarks, 2), copy=False)



