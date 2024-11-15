from .util import *
from .bluestein_fft import zoom_fft_2d_mod
from .data import telescope_params
import numpy as np

#If want to do planets are stars separately, call starshade prop twice or something

class StarshadeProp:
    """
    drm : 'wfirst' or 'hwo'
    """
    def __init__(self, source_field, drm = None, ):
        self.light_source = light_source()  # Instantiate the light source
        self.detector = detector()          # Instantiate the detector
        self.lens = lens() if lens else None  # Optionally include a lens
        self.d_p_mas = ..
        factor_source_pupil = d_s_mas/d_p_mas 
        if np.abs(factor_source_pupil%1):
            from scipy.ndimage import zoom:
            scale_factor = d_x / d_x_new

            # Apply zoom to resample the image with the calculated scale factor
            new_image = zoom(image, scale=(scale_factor, scale_factor), order=3)

        if drm is not None: 
            
            drm_params = telescope_params[drm]



    # a wl range has to be defined
    # for a given source field, either grab f name of or generate the PSF over wavelengths
    # one function to convolve the source with this over a specific wl
    # then             
    def source_convolve:
        #test = np.zeros((2*N_t - 1, 2*N_t - 1))
        #test[100, 0] = 1
        # need to stagger the source field by factor_source_pupil
        # what if its a non-int?

        source_upsampled = np.zeros(())
        source = bluestein_pad(source_field, )
        #test2 = bluestein_pad(field_free_prop, N_t, N_t)
        #test3 = np.fft.ifft2(np.fft.fft2(test)*np.fft.fft2(test2))
        #print (np.allclose(field_free_prop, test3[N_t - (N_t//2) + 100: N_t + N_t//2 + 1 + 100, N_t - (N_t//2) : N_t + N_t//2 + 1]))


    def run_simulation(self):
        # Example method that might coordinate interactions between components
        light = self.light_source.emit()

    def pupilfield()
        """
        Generates the incoherent field at the pupil for choice of starshade
        """
        if os.path.exists('pupil_out/hwo_pupil_'+drm_params['grey_mask_dx'][mask_choice]+'_'+str(500)+'.npz'):
            print("File exists.")
        else:
            print("File does not exist.")
            for wl_i in np.arange(500, 1050, 50, dtype=np.float64):
                 print (wl_i)
                 chunk_in_source_field_to_pupil(source_field, wl_i*1e-09, dist_xo_ss, dist_ss_t, ss_mask_fname_memmap, N_s = 101, N_x = N_x, N_t = over_N_t, ds = 0.04*au_to_meter,  dx = drm_params['dx_'][mask_choice], dt = dt)


class SourceField:
    """
    This class returns far-field diffraction for an N_s x N_s source-field, calculated via a Bluestein FFT
    Either define d_s_mas or d_s and z

    Attributes:
        d_s_mas (float): Source field sampling in mas
        d_s (float): Source field sampling
        N_s (int): Number of samples of the field
        wavelength (float): Wavelength of light
        z (float): Far-field distance
    Note: Uses planar wave description of far-field
    """
    def __init__(self, N_s, wavelength, source_field, d_s_mas = None, d_s = None, z = None):
        self.N_s = N_s
        self.wavelength = wavelength
        self.d_s = d_s if d_s is not None
        self.z = z if z is not None
        self.d_s_mas = d_s_mas if d_s_mas is not None else self.calculate_d_s_mas(self.d_s, self.z)
        self.source_field = source_field
        if any(f.startswith("data/"+ss_file) for f in os.listdir()):

    def calculate_d_s_mas(self, d_s, z):
        return (d_s / z) * rad_to_mas

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
        out_field = zoom_fft_2d_mod(source_field, self.N_s, N_x, ZP)
        out_fac = np.exp( 1j* (2 * np.pi/self.wavelength) * z1)
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
