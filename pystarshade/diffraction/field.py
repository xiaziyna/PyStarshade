from diffraction.util import *
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

    def update(self, d_x=None, N=None):
        """
        Update the values of d_x, N, and z.

        Args:
            d_x (float, optional): New value for d_x.
            N (int, optional): New value for N.
            z (float, optional): New value for z.
        """
        if d_x is not None:
            self.d_x = d_x

        if N is not None:
            self.N = N

    def wave_numbers(self):
        """
        Calculate the planar wave numbers for a point source at position (x, y, z)

        See Blahut (Theory of Remote Image Formation, sec 1.7).

        Args: 
            Coordinates of source (x, y, z)

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
        return k

    def plane_wave(self, k, z1):
        """
        Calculate a plane wave with amplitude A and wavevector k on a Cartesian grid.
        
        Arg:
            k: Wave vector 
            z1: Propagtion distance (Note: for computational purposes treat origin as starshade plane)

        The grid has a pixel size of d_x and dimensions N x N. The resulting plane wave is 
        computed for each point on the grid.

        Returns:
            numpy.ndarray: A 2D array representing the computed plane wave over the grid.
        """     
        vals = np.linspace(-(self.N//2),(self.N//2),self.N)*self.d_x
        xx, yy = np.meshgrid(vals, vals)
        vec = np.array([xx,yy]).T
        return self.A*np.exp(1j* np.inner(k[:2], vec))*np.exp(1j * k[2]*z1) 
