import numpy as np


class ImageCube:
    """Class to make a 3D cube."""

    def __init__(self, image2D, tExp=120):
        self.image2D = image2D
        self.imageCube = self.makeImageCube
        self.tFrame = 3.08
        self.tExp = tExp
        self.nFrames = self.tExp // self.tFrame

    def makeImageCube(self):
        """Makes a cube."""
        rng = np.random.default_rng()
        pArray = [1 / (self.nFrames - 1)] * (self.nFrames - 1)
        rvs = rng.multinomial(self.image2D, pArray, size=(self.nFrames - 1,) + self.image2D.shape)
        return np.cumsum(rvs)
