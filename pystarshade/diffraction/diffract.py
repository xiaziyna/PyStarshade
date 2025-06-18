import numpy as np
from pystarshade.diffraction.bluestein_fft import zoom_fft_2d_mod, zoom_fft_2d, chunk_in_chirp_zoom_fft_2d_mod, zoom_fft_2d_cached
from pystarshade.diffraction.util import *

class Fresnel:
    """
    Represents a Fresnel diffraction computation setup.

    Attributes
    ----------
    d_x : float
        Spatial sampling interval of the input field [m].
    N_in : int
        Number of non-zero input samples.
    z : float
        Propagation distance [m].
    wavelength : float
        Wavelength of the light [m].
    wl_z : float
        Product of the wavelength and propagation distance.
    k : float
        Wave number.
    max_freq : float
        Maximum frequency of the input field.
    """
    def __init__(self, d_x, N_in, z, wavelength):
        self.d_x = d_x
        self.N_in = N_in
        self.z = z
        self.wavelength = wavelength
        self.wl_z = self.wavelength * self.z
        self.k = 2 * np.pi / self.wavelength
        self.max_freq = 1 / self.d_x

class FresnelSingle(Fresnel):
    """
    Single FFT Fresnel diffraction.

    Attributes
    ----------
    d_x : float
        Spatial sampling interval of the input field [m].
    d_f : float
        Desired frequency sampling interval [m^-1].
    N_in : int
        Number of non-zero input samples.
    z : float
        Propagation distance [m].
    wavelength : float
        Wavelength of the light [m].
    wl_z : float
        Product of the wavelength and propagation distance.
    ZP : float
        Zero-padding factor.
    N_X : float
        Phantom length for zero-padded signal.

    Inherits From
    -------------
    Fresnel
    """
    def __init__(self, d_x, d_f, N_in, z, wavelength):
        super().__init__(d_x, N_in, z, wavelength)
        self.d_f = d_f
        self.ZP = self.calc_zero_padding()
        self.N_X = self.calc_phantom_length()

    def calc_phantom_length(self):
        """
        Calculate the equivalent zero-padded signal length N_X to achieve a specified output 
        frequency (d_f) for spatial sampling (d_x) using the single FT method.
    

        Returns
        -------
        float
            Zero-padded signal length (N_X). N_X = Z_pad * N_in + 1 - phantom padded length of input signal
        """
        N_X = self.max_freq * self.wl_z / self.d_f
        return N_X

    def calc_zero_padding(self):
        """
        Calculate the zero-padding factor (ZP * N_in) to achieve a specified frequency spacing
        (d_f) for a given spatial sampling (d_x) using the Fresnel single FT method.

        Returns
        -------
        float
             Zero-padding factor (ZP * N_in) to achieve the desired frequency spacing.
        """


        ZP = ((self.max_freq * self.wl_z / self.d_f) - 1) / self.N_in
        return ZP

    def zoom_fresnel_single_fft(self, field, N_out):
        """
        Perform single FFT Fresnel diffraction using a Bluestein FFT.
        Output on grid defined by (N_out/ZP*N_in)*wl_z/d_x, (N_out/ZP*N_in)*wl_z/d_x

        Parameters
        ----------
        field : np.ndarray
            2D input field to be propagated.
        N_out : int
            Number of output samples in each dimension.

        Returns
        -------
        tuple
            - np.ndarray: Propagated output field.
            - float: Output grid sampling.
        """
        Ny, Nx = field.shape
        in_xy = grid_points(Nx, Ny, dx = self.d_x)
        field *= np.exp(1j * (np.pi /self.wl_z) * (in_xy[0]**2 + in_xy[1]**2))
        output_field = zoom_fft_2d_mod(field, self.N_in, N_out, N_X=self.N_X) * (self.d_x**2)
        df = self.max_freq * self.wl_z / self.N_X
#        output_field = zoom_fft_2d_mod(field, self.N_in, N_out, Z_pad=self.ZP) * (self.d_x**2)
#        df = (self.max_freq*self.wl_z / (self.ZP*self.N_in + 1))
        out_xy = grid_points(N_out, N_out, dx = df )
        quad_out_fac = np.exp(1j * self.k * self.z) * np.exp(1j * self.k / (2 * self.z) * (out_xy[0]**2 + out_xy[1]**2)) / ( 1j * self.wl_z) 
        return quad_out_fac * output_field, df

    def nchunk_zoom_fresnel_single_fft(self, x_file, N_out, N_chunk = 4):
        """
        Single FFT Fresnel diffraction calculated using an N_chunk*N_chunk -way chunked
        Bluestein FFT (caps peak memory usage). Use me if the mask is big!

        Define x_file as: 
        arr = np.memmap('x.dat', dtype=np.complex128,mode='w+',shape=(N_x, N_x))
        arr[:] = x
        arr.flush()

        Parameters
        ----------
        x_file : np.memmap
            Input mask filename as a memory-mapped object.
        N_out : int
            Number of output samples.
        N_chunk : int, optional
            Number of chunks. Default is 4.

        Returns
        -------
        tuple
            - np.ndarray: Propagated output field.
            - float: Output grid sampling.
        """
        field = chunk_in_chirp_zoom_fft_2d_mod(x_file, self.wl_z, self.d_x, self.N_in, N_out, self.N_X, N_chunk = 4) * (self.d_x**2)
        df = self.max_freq*self.wl_z / self.N_X
        out_xy = grid_points(N_out, N_out, dx = df)
        quad_out_fac = np.exp(1j * self.k * self.z) * np.exp(1j * self.k / (2 * self.z) * (out_xy[0]**2 + out_xy[1]**2)) / ( 1j * self.wl_z)
        return quad_out_fac * field, df

    def chunk_zoom_fresnel_single_fft(self, x_file, N_out):
        """
        Single FFT Fresnel diffraction calculated using a four-way chunked
        Bluestein FFT (caps peak memory usage).

        Args
        ----
        x_file : np.memmap
            Input mask filename as a memmap object.
        N_out : int
            Number of output samples.

        Returns
        -------
        tuple
            - output_field : np.ndarray
                Diffracted output field.
            - df : float
                Sample size.
        """
        bit_x = self.N_in%2
        for chunk in range(4):
            x_trunc = np.memmap(x_file, dtype=np.complex128, mode='w+', shape=(self.N_in, self.N_in))
            if chunk == 0:
                xx = np.arange(-(self.N_in//2), 1)[np.newaxis, :] * self.d_x
                yy = np.arange(-(self.N_in//2), 1)[:, np.newaxis] * self.d_x
                x_trunc[:self.N_in//2 + bit_x, :self.N_in//2 + bit_x] *= np.exp(1j * (np.pi /wl_z) * (xx**2 + yy**2)).astype(np.complex128)
            elif chunk == 1:
                xx = np.arange(-(self.N_in//2), 1)[:, np.newaxis] * self.d_x
                yy = np.arange(1, self.N_in//2 + bit_x)[np.newaxis, :] * self.d_x
                x_trunc[:self.N_in//2 + bit_x, self.N_in//2 + bit_x:] *= np.exp(1j * (np.pi /wl_z) * (xx**2 + yy**2)).astype(np.complex128)
            elif chunk == 2:
                xx = np.arange(1, self.N_in//2 + bit_x)[np.newaxis, :] * self.d_x
                yy = np.arange(1, self.N_in//2 + bit_x)[:, np.newaxis] * self.d_x
                x_trunc[self.N_in//2 + bit_x:, self.N_in//2 + bit_x:] *= np.exp(1j * (np.pi /wl_z) * (xx**2 + yy**2)).astype(np.complex128)
            elif chunk == 3:
                xx = np.arange(1, self.N_in//2 + bit_x)[:, np.newaxis] * self.d_x
                yy = np.arange(-(self.N_in//2), 1)[np.newaxis, :] * self.d_x
                x_trunc[self.N_in//2 + bit_x:, :self.N_in//2 + bit_x] *= np.exp(1j * (np.pi /wl_z) * (xx**2 + yy**2)).astype(np.complex128)
            x_trunc.flush()
        output_field = four_chunked_zoom_fft_mod(x_file, self.N_in, N_out, self.N_X) * (self.d_x**2)
        df = self.max_freq*self.wl_z / self.N_X
        out_xy = grid_points(N_out, N_out, dx = df )
        quad_out_fac = np.exp(1j * self.k * self.z) * np.exp(1j * self.k / (2 * self.z) * (out_xy[0]**2 + out_xy[1]**2)) / ( 1j * self.wl_z)
        return quad_out_fac * output_field, df

    def one_chunk_zoom_fresnel_single_fft(self, field, N_out, chunk=0):
        """
        Perform single FFT Fresnel diffraction on a single quadrant.

        Use this method if you'd rather not load a full-starshade or store one in memory.
        First, call `zoom_fft_quad_out_mod` and compute the field on a single quadrant.

        Parameters
        ----------
        field : np.ndarray
            Input field for the computation.
        N_out : int
            Number of output samples.
        chunk : int, optional
            Quadrant index (0-3). Default is 0.

        Returns
        -------
        tuple
            - field : np.ndarray
                The propagated output field for the specified quadrant.
            - df : float
                Output grid sampling.
        """
        k = 2 * np.pi / self.wavelength
        bit_x = self.N_in%2
        if chunk == 0:
            x = np.arange(-(self.N_in//2), 1)[np.newaxis, :] * self.d_x
            y = np.arange(-(self.N_in//2), 1)[:, np.newaxis] * self.d_x
        elif chunk == 1:
            x = np.arange(-(self.N_in//2), 1)[:, np.newaxis] * self.d_x
            y = np.arange(1, self.N_in//2 + bit_x)[np.newaxis, :] * self.d_x
        elif chunk == 2:
            x = np.arange(1, self.N_in//2 + bit_x)[np.newaxis, :] * self.d_x
            y = np.arange(1, self.N_in//2 + bit_x)[:, np.newaxis] * self.d_x
        elif chunk == 3:
            x = np.arange(1, self.N_in//2 + bit_x)[:, np.newaxis] * self.d_x
            y = np.arange(-(self.N_in//2), 1)[np.newaxis, :] * self.d_x
        field *= np.exp(1j * (np.pi /self.wl_z) * (x**2 + y**2))
        field = single_chunked_zoom_fft_mod(field, self.N_in, N_out, self.N_X, i=chunk) * (self.d_x**2)
        df = self.max_freq*self.wl_z / self.N_X
        out_xy = grid_points(N_out, N_out, dx = df )
        quad_out_fac = np.exp(1j * k * self.z) * np.exp(1j * k / (2 * self.z) * (out_xy[0]**2 + out_xy[1]**2)) / ( 1j * self.wl_z)
        return quad_out_fac * field, df

class FresnelDouble(Fresnel):
    """
    This class inherits from the `Fresnel` class and represents a setup for
    computing Fresnel diffraction where the output spatial grid matches the
    input spatial grid i.e. double Fourier transform Fresnel diffraction.

    Parameters
    ----------
    d_x : float
        Spatial sampling interval of the input field [m].
    N_in : int
        Number of non-zero input samples.
    z : float
        Propagation distance [m].
    wavelength : float
        Wavelength of light [m].

    Inherits From
    -------------
    Fresnel
    """

    def __init__(self, d_x, N_in, z, wavelength):
        super().__init__(d_x, N_in, z, wavelength)

    def fresnel_double_fft(self, field):
        """
        Perform double FFT Fresnel diffraction.

        The output is computed on the same grid as the input field.

        Parameters
        ----------
        field : np.ndarray
            2D input field to be propagated.

        Returns
        -------
        tuple
            - output_field : np.ndarray
                Propagated output field.
            - df : float
                Output grid sampling.
        """
        k = 2 * np.pi / self.wavelength
        Ny, Nx = field.shape
        df = self.max_freq / Nx
        freq_xy = grid_points(Nx, Ny, dx = df)
        spectral_field =  np.fft.fft2(field) * np.exp(1j * self.z * k)\
         * np.exp(-1j * np.pi * self.wl_z * np.fft.fftshift(freq_xy[0]**2 + freq_xy[1]**2) ) 
        output_field = np.fft.ifft2(spectral_field)

        return output_field, df

class Fraunhofer:
    """
    This class computes Fraunhofer diffraction, including zero-padding calculations.

    Attributes
    ----------
    d_x : float
        Spatial sampling interval of the input field [m].
    d_f : float
        Desired frequency sampling interval [m^-1].
    N_in : int
        Number of non-zero input samples.
    z : float
        Propagation distance [m].
    wavelength : float
        Wavelength of light [m].
    wl_z : float
        Product of the wavelength and propagation distance (wavelength * z).
    max_freq : float
        Maximum frequency of the input field, defined as 1 / d_x.
    ZP : float
        Zero-padding factor, computed to achieve a specified frequency spacing.
    N_X : float
        Phantom length for zero-padded signal, calculated based on input parameters.
    """
    def __init__(self, d_x, d_f, N_in, z, wavelength):
        self.d_x = d_x
        self.d_f = d_f
        self.N_in = N_in
        self.z = z
        self.wavelength = wavelength
        self.wl_z = self.wavelength * self.z
        self.max_freq = 1 / self.d_x
        self.ZP = self.calc_zero_padding()
        self.N_X = self.calc_phantom_length()

    def calc_phantom_length(self):
        """
        Calculate the equivalent zero-padded signal length (N_X) to achieve a specified output
        frequency (d_f) for spatial sampling (d_x) using the single FT method.

        Parameters
        ----------
        None (all relevant attributes are already part of the class).

        Returns
        -------
        float
            N_X :  N_X = Z_pad * N_in + 1 - phantom padded length of input signal
        """

        N_X = self.max_freq * self.wl_z / self.d_f
        return N_X
        
    def calc_zero_padding(self):
        """
        Calculate the zero-padding factor (ZP * N_in) to achieve a specified frequency spacing
        (d_f) for a given spatial sampling (d_x) using the Fresnel single FT method.

        Parameters
        ----------
        None (all relevant attributes are already part of the class).

        Returns
        -------
        float
            Zero-padding factor (ZP * N_in) to achieve the desired frequency spacing.
        """

        ZP = ((self.max_freq * self.wl_z / self.d_f) - 1) / self.N_in
        return ZP

    def zoom_fraunhofer(self, field, N_out):
        """
        Perform Fraunhofer diffraction at points within
        [(N_out/ZP*N_in)*wl_z/d_x, (N_out/ZP*N_in)*wl_z/d_x] using the Bluestein FFT.

        Parameters
        ----------
        field : np.ndarray
            2D input field to be propagated.
        N_out : int
            Number of output samples in each dimension.

        Returns
        -------
        tuple
            - output_field : np.ndarray
                Propagated output field.
            - df : float
                Output grid sampling.
        """
        k = 2 * np.pi / self.wavelength
        Ny, Nx = field.shape    
        output_field = zoom_fft_2d_cached(field, self.N_in, N_out, N_X = self.N_X) * (self.d_x**2)
        df = self.max_freq*self.wl_z / self.N_X
        out_xy = grid_points(N_out, N_out, dx = df )
        out_fac = np.exp ( ( 1j * k / (2 * self.z) ) * (out_xy[0]**2 + out_xy[1]**2) ) / (1j * self.wl_z)

        return out_fac*output_field, df
