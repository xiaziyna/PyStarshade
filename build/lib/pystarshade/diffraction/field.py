from .util import *
from .bluestein_fft import zoom_fft_2d_mod
import numpy as np

class SourceField:
    """
    This class returns far-field diffraction for an N_s x N_s source-field, calculated via a Bluestein FFT
    
    Attributes:
        d_s (float): Source field sampling
        N_s (int): Number of samples of the field
        wavelength (float): Wavelength of light
        z (float): Far-field distance
    Note: Uses planar wave description of far-field
    """
    def __init__(self, d_s, N_s, wavelength, z, source_field):
        self.d_s = d_s
        self.N_s = N_s
        self.wavelength = wavelength
        self.z = z
        self.source_field = source_field

    def farfield(self, d_x, N_x, z1):
        """
        Calculate the far-field diffraction of source field at local distance z1 over (N_x, N_x) grid with sampling dx
        Note: z1 is defined as a local distance, the origin is set in the far-field.

        Arg:
            d_x (float): Output spatial sampling 
            N_x (int): Number of output samples
            z1 (float): Propagtion distance (Note: for computational purposes treat origin as starshade plane)

        Returns:
            numpy.ndarray: A 2D array representing the computed field over the grid.
        """ 
        ZP = (((1/self.d_s) * self.wavelength * self.z / d_x) - 1) / self.N_s
        source_field_pad = bluestein_pad(self.source_field, self.N_s, N_x)
        out_field = zoom_fft_2d_mod(source_field_pad, self.N_s, N_x, ZP)
        out_fac = np.exp ( 1j * 2*np.pi * z1 / self.wavelength) 
        return out_fac*out_field

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

    def update(self, d_x=None, N=None):
        """
        Update the values of d_x, N, and z.

        Args:
            d_x (float, optional): New value for d_x.
            N (int, optional): New value for N.
        """
        if d_x is not None:
            self.d_x = d_x
        if N is not None:
            self.N = N

class GaussianSource(Field):
    """
    A class to calculate the far-field of a Gaussian source.

    Attributes:
        d_x (float): Spatial sampling of the output field.
        N (int): Number of samples of the output field.
        wavelength (float): Wavelength of light.
        x (float): The x-coordinate of the Gaussian source origin in the source field.
        y (float): The y-coordinate of the Gaussian source origin in the source field.
        z (float): Far-field distance.
        A (float): The amplitude of the Gaussian source.
        sigma (float): The standard deviation of the Gaussian source.


    Inherits from:
        Field

    Note: source field is given by A * np.exp(-(x/sigma)**2)
    """

    def __init__(self, d_x, N, wavelength, x, y, z, A, sigma):
        super().__init__(d_x, N, wavelength, z)
        self.x = x        
        self.y = y
        self.A = A
        self.sigma = sigma
        self.wl_z = self.wavelength * self.z


    def far_field_gaussian_params(self):
        """
        Calculates the far-field amplitude (A) and inverse standard deviation (1/sigma).

        Returns:
            far_A (float): The far-field amplitude.
            far_inv_sigma (float): The inverse standard deviation of the far-field Gaussian.

        Notes: See https://www.pas.rochester.edu/~dmw/ast203/Lectures/Lect_14.pdf
        """
        far_A = self.A * np.pi * self.sigma**2 / self.wl_z
        far_inv_sigma = np.pi * self.sigma / self.wl_z
        return far_A, far_inv_sigma

    def far_field_gaussian(self, far_A, far_inv_sigma, z1):
        """
        Calculates the far-field Gaussian with parameters (far_A, 1/far_inv_sigma) on a Cartesian grid. 

        Args:
            far_A (float): The far-field amplitude.
            far_inv_sigma (float): The inverse standard deviation of the far-field Gaussian.
            z1 (float): Distance (Note: for computational purposes treat origin as starshade plane).

        Returns:
            numpy.ndarray: A 2D array representing the far-field Gaussian evaluated on a Cartesian grid.
        """
        vals = np.linspace(-(self.N//2),(self.N//2),self.N)*self.d_x
        xx, yy = np.meshgrid(vals, vals)

        return far_A * np.exp( - ( xx**2 + yy**2 ) * far_inv_sigma**2) * np.exp( 1j * 2 * np.pi * z1 / self.wavelength)

class PointSource(Field):
    def __init__(self, d_x, N, wavelength, x, y, z, A):
        super().__init__(d_x, N, wavelength, z)
        self.x = x        
        self.y = y
        self.A = A

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
