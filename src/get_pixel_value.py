import numpy as np
from menpo.image.interpolation import scipy_interpolation
from abc import abstractmethod
from base import FeatureExtractor, FeatureExtractorBuilder
    
# Get pixel value from image

class GetPixelValueBuild(FeatureExtractorBuilder):
    def __init__(self,n_landmarks,n_pixels,kappa):
        self.n_landmarks = n_landmarks
        self.n_pixels = n_pixels
        self.kappa = kappa
        
    def build(self):
        return GetPixelValue(self.n_landmarks,self.n_pixels,self.kappa)
        

class GetPixelValue(FeatureExtractor):
    def __init__(self,n_landmarks,n_pixels,kappa):
        self.lmark = np.random.randint(low=0, high=n_landmarks, size=n_pixels)
        self.pixel_delta = np.random.uniform(low=-kappa, high=kappa, size=n_pixels*2).reshape(n_pixels, 2)
        
    def get_feature(self, img, shape, mean_to_shape):
        pixel_coords = shape.points[self.lmark]+mean_to_shape.apply(self.pixel_delta)
        ret = scipy_interpolation(img.pixels, pixel_coords, order=1, mode='constant', cval=0)
        return ret.reshape((len(pixel_coords),))
    