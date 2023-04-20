from util import *
import numpy as np

class Field:
    """
    This class returns analytic far-field diffraction for sources, or can be used to generate source fields
    Inlcudes : Point source (off-axis), planar fields, Gaussian sources
    
    Attributes:
        d_x (float): Spatial sampling of the field
        N (int): Number of samples of the field
        wavelength (float): Wavelength of light
        z (float): Far-field distance

    Note:
        Any additional notes or important information about the class.
    """
    def __init__(self, d_x, N, wavelength, z):
        self.d_x = d_x
        self.N = N
        self.wavelength = wavelength
        self.z = z

class PointSource(Field):
    def __init__(self, d_x, N, wavelength, x, y, z, A):
        super().__init__(d_x, N, wavelength, z)
        self.x = x        
        self.y = y
        self.A = A
        self.k = self.point_source_planar_wave_numbers()

    def point_source_planar_wave_numbers(self):
        """
        Calculate the planar wave numbers for a point source at position (x, y, z)

        See Blahut (Theory of Remote Image Formation, sec 1.7).

        Args: 
            Coordinates (x, y, z)

        Returns:
            numpy.ndarray: Planar wave numbers [k_x, k_y, k_z] for the point source.
        """
        if (self.x == 0 and self.y == 0): 
            theta = 0
        else: theta = np.hypot(self.x, self.y) / self.z
        if self.x == 0:
            phi = np.pi/2
        else: phi = np.arccos ( self.x / np.hypot(self.x, self.y) )
        ax = np.cos(phi) * np.sin(theta)
        by = np.sin(theta) * np.sin(phi)
        cz = np.cos(theta)
        k = np.array([ax, by, cz]) * 2*np.pi/self.wavelength
        print (k)
        return k

    def plane_wave(self):
        """
        Calculate a plane wave with amplitude A and wavevector k on a Cartesian grid.

        The grid has a pixel size of d_x and dimensions N x N. The resulting plane wave is 
        computed for each point on the grid.

        Returns:
            numpy.ndarray: A 2D array representing the computed plane wave over the grid.
        """     
        vals = np.linspace(-(self.N//2),(self.N//2),self.N)*self.d_x
        xx, yy = np.meshgrid(vals, vals)
        vec = np.array([xx,yy,np.zeros((self.N, self.N))]).T
        return self.A*np.exp(1j* np.inner(self.k, vec))    
