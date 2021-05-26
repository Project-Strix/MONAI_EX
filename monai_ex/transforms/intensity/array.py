"""
A collection of "vanilla" transforms for intensity adjustment
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from typing import Any, Optional, Sequence, Union

import numpy as np
from scipy.special import comb
from scipy.ndimage import median_filter

from monai.transforms.compose import Randomizable, Transform


class ClipIntensity(Transform):
    """Clip intensity by given range

    Args:
        cmin (float): intensity target range min.
        cmax (float): intensity target range max.
    """
    def __init__(self, cmin: float, cmax: float) -> None:
        self.cmin = cmin
        self.cmax = cmax

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = np.clip(img, self.cmin, self.cmax)
        return img


class RandLocalPixelShuffle(Randomizable, Transform):
    def __init__(self, prob: float = 0.5, num_block_range: Union[Sequence[int], int] = [50,200]):
        self.num_block_range = (num_block_range,num_block_range+1) if isinstance(num_block_range, int) else num_block_range
        self.prob = min(max(prob, 0.0), 1.0)

    def randomize(self, data: Optional[Any] = None) -> None:
        shape = data.squeeze().shape
        self.num_block = self.R.randint(self.num_block_range[0], self.num_block_range[1], 1)[0]
        if len(shape) == 3:
            self.img_rows, self.img_cols, self.img_deps = shape
            self.dim = 3
        elif len(shape) == 2:
            self.img_rows, self.img_cols = shape
            self.dim = 2
        else:
            raise ValueError('Only support 2D and 3D images')
        self._do_transform = self.R.random() < self.prob

    def generate_pos(self):
        self.block_noise_size_x = self.R.randint(1, self.img_rows//10)
        self.block_noise_size_y = self.R.randint(1, self.img_cols//10)
        self.block_noise_size_z = self.R.randint(1, self.img_deps//10) if self.dim ==3 else None
        self.noise_x = self.R.randint(0, self.img_rows-self.block_noise_size_x)
        self.noise_y = self.R.randint(0, self.img_cols-self.block_noise_size_y)
        self.noise_z = self.R.randint(0, self.img_deps-self.block_noise_size_z) if self.dim == 3 else None

    def __call__(self, image):
        self.randomize(image)
        if not self._do_transform:
            return image
        
        image_temp = image.copy()
        for _ in range(self.num_block):
            self.generate_pos()
            if self.dim == 3:
                window = image[0, 
                             self.noise_x:self.noise_x+self.block_noise_size_x, 
                             self.noise_y:self.noise_y+self.block_noise_size_y, 
                             self.noise_z:self.noise_z+self.block_noise_size_z,
                            ]
            elif self.dim == 2:
                window = image[0, 
                             self.noise_x:self.noise_x+self.block_noise_size_x, 
                             self.noise_y:self.noise_y+self.block_noise_size_y,
                            ]
            window = window.flatten()
            np.random.shuffle(window)
            if self.dim == 3:
                window = window.reshape((self.block_noise_size_x, self.block_noise_size_y, self.block_noise_size_z))
                image_temp[0, 
                           self.noise_x:self.noise_x+self.block_noise_size_x, 
                           self.noise_y:self.noise_y+self.block_noise_size_y, 
                           self.noise_z:self.noise_z+self.block_noise_size_z ] = window
            elif self.dim == 2:
                window = window.reshape((self.block_noise_size_x, self.block_noise_size_y))
                image_temp[0, 
                           self.noise_x:self.noise_x+self.block_noise_size_x, 
                           self.noise_y:self.noise_y+self.block_noise_size_y ] = window

        return image_temp


class RandImageInpainting(Randomizable, Transform):
    def __init__(self, prob: float = 0.5, num_block_range: Union[Sequence[int], int] = [3,6]):
        self.num_block_range = (num_block_range,num_block_range+1) if isinstance(num_block_range, int) else num_block_range
        self.prob = min(max(prob, 0.0), 1.0)

    def randomize(self, data: Optional[Any] = None) -> None:
        self._do_transform = self.R.random() < self.prob
        self.num_block = self.R.randint(self.num_block_range[0], self.num_block_range[1], 1)[0]
        shape = data.squeeze().shape
        if len(shape) == 3:
            self.img_rows, self.img_cols, self.img_deps = shape
            self.dim = 3
        elif len(shape) == 2:
            self.img_rows, self.img_cols = shape
            self.dim = 2
        else:
            raise ValueError('Only support 2D and 3D images')

    def generate_pos(self):
        self.block_noise_size_x = self.R.randint(self.img_rows//6, self.img_rows//3)
        self.block_noise_size_y = self.R.randint(self.img_cols//6, self.img_cols//3)
        self.block_noise_size_z = self.R.randint(self.img_deps//6, self.img_deps//3) if self.dim ==3 else None
        self.noise_x = self.R.randint(3, self.img_rows-self.block_noise_size_x-3)
        self.noise_y = self.R.randint(3, self.img_cols-self.block_noise_size_y-3)
        self.noise_z = self.R.randint(3, self.img_deps-self.block_noise_size_z-3) if self.dim ==3 else None

    def __call__(self, image):
        self.randomize(image)
        if not self._do_transform:
            return image

        for _ in range(self.num_block):
            self.generate_pos()
            if self.dim == 3:
                image[:,
                      self.noise_x:self.noise_x+self.block_noise_size_x,
                      self.noise_y:self.noise_y+self.block_noise_size_y,
                      self.noise_z:self.noise_z+self.block_noise_size_z] = np.random.rand(self.block_noise_size_x,
                                                                                          self.block_noise_size_y,
                                                                                          self.block_noise_size_z ) * 1.0
            elif self.dim == 2:
                image[:,
                      self.noise_x:self.noise_x+self.block_noise_size_x,
                      self.noise_y:self.noise_y+self.block_noise_size_y] = np.random.rand(self.block_noise_size_x,
                                                                                          self.block_noise_size_y ) * 1.0                
        return image


class RandImageOutpainting(Randomizable, Transform):
    def __init__(self, prob: float = 0.5, num_block_range: Union[Sequence[int], int] = [3,6]):
        self.num_block_range = (num_block_range,num_block_range+1) if isinstance(num_block_range, int) else num_block_range
        self.prob = min(max(prob, 0.0), 1.0)

    def randomize(self, data: Optional[Any] = None) -> None:
        self._do_transform = self.R.random() < self.prob
        self.num_block = self.R.randint(self.num_block_range[0], self.num_block_range[1], 1)[0]
        shape = data.squeeze().shape
        if len(shape) == 3:
            self.img_rows, self.img_cols, self.img_deps = shape
            self.dim = 3
        elif len(shape) == 2:
            self.img_rows, self.img_cols = shape
            self.dim = 2
        else:
            raise ValueError('Only support 2D and 3D images')

    def generate_pos(self):
        ratio = 8
        self.block_noise_size_x = self.img_rows - self.R.randint(3*self.img_rows//ratio, 4*self.img_rows//ratio)
        self.block_noise_size_y = self.img_cols - self.R.randint(3*self.img_cols//ratio, 4*self.img_cols//ratio)
        self.block_noise_size_z = self.img_deps - self.R.randint(3*self.img_deps//ratio, 4*self.img_deps//ratio) if self.dim ==3 else None
        self.noise_x = self.R.randint(3, self.img_rows-self.block_noise_size_x-3)
        self.noise_y = self.R.randint(3, self.img_cols-self.block_noise_size_y-3)
        self.noise_z = self.R.randint(3, self.img_deps-self.block_noise_size_z-3) if self.dim ==3 else None

    def __call__(self, image):
        self.randomize(image)
        if not self._do_transform:
            return image

        self.generate_pos()
        image_temp = image.copy()
        x = self.R.rand(*image.shape) * 1.0
        if self.dim == 3:
            x[:, 
              self.noise_x:self.noise_x+self.block_noise_size_x, 
              self.noise_y:self.noise_y+self.block_noise_size_y, 
              self.noise_z:self.noise_z+self.block_noise_size_z] = image_temp[:, 
                                                                              self.noise_x:self.noise_x+self.block_noise_size_x, 
                                                                              self.noise_y:self.noise_y+self.block_noise_size_y, 
                                                                              self.noise_z:self.noise_z+self.block_noise_size_z]
        elif self.dim == 2:
            x[:, 
              self.noise_x:self.noise_x+self.block_noise_size_x, 
              self.noise_y:self.noise_y+self.block_noise_size_y] = image_temp[:, 
                                                                              self.noise_x:self.noise_x+self.block_noise_size_x, 
                                                                              self.noise_y:self.noise_y+self.block_noise_size_y]
        for _ in range(self.num_block):
            self.generate_pos()
            if self.dim == 3:
                x[:, 
                  self.noise_x:self.noise_x+self.block_noise_size_x, 
                  self.noise_y:self.noise_y+self.block_noise_size_y, 
                  self.noise_z:self.noise_z+self.block_noise_size_z] = image_temp[:, 
                                                                                  self.noise_x:self.noise_x+self.block_noise_size_x, 
                                                                                  self.noise_y:self.noise_y+self.block_noise_size_y, 
                                                                                  self.noise_z:self.noise_z+self.block_noise_size_z]
            elif self.dim == 2:
                x[:, 
                  self.noise_x:self.noise_x+self.block_noise_size_x, 
                  self.noise_y:self.noise_y+self.block_noise_size_y] = image_temp[:, 
                                                                                  self.noise_x:self.noise_x+self.block_noise_size_x, 
                                                                                  self.noise_y:self.noise_y+self.block_noise_size_y]
        return x


class RandNonlinear(Randomizable, Transform):
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def randomize(self, data: Optional[Any] = None) -> None:
        self._do_transform = self.R.random() < self.prob
        self.points = [[0, 0], [self.R.random(), self.R.random()], [self.R.random(), self.R.random()], [1, 1]]
    
    def bernstein_poly(self, i, n, t):
        """
        The Bernstein polynomial of n, i as a function of t
        """

        return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

    def bezier_curve(self, points, nTimes=10000):
        """
        Given a set of control points, return the
        bezier curve defined by the control points.
        Control points should be a list of lists, or list of tuples
        such as [ [1,1], 
                    [2,3], 
                    [4,5], ..[Xn, Yn] ]
            nTimes is the number of time steps, defaults to 1000
            See http://processingjs.nihongoresources.com/bezierinfo/
        """

        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, nTimes)

        polynomial_array = np.array([ self.bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
        
        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)

        return xvals, yvals

    def __call__(self, image):
        self.randomize(image)
        if not self._do_transform:
            return image

        xpoints = [p[0] for p in self.points]
        ypoints = [p[1] for p in self.points]
        xvals, yvals = self.bezier_curve(self.points, nTimes=10000)
        if self.R.random() < 0.5:
            # Half change to get flip
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        
        nonlinear_x = np.interp(image, xvals, yvals)
        return nonlinear_x


class MedianFilter(Transform):
    """Calculate a multidimensional median filter.
       Wrapper of scipy.ndimage.median_filter.
    """
    def __init__(self, size: int, mode='reflect', cval=0.0, origin=0) -> None:
        self.size = size
        self.mode = mode
        self.cval = cval
        self.origin = origin

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = median_filter(img, size=self.size, mode=self.mode, cval=self.cval, origin=self.origin)
        return img
