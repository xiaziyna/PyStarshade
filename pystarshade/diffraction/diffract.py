import numpy as np
from bluestein_fft import zoom_fft_2d_mod, zoom_fft_2d, chunk_in_chirp_zoom_fft_2d_mod
from util import *

class Fresnel:
    def __init__(self, d_x, N_in, z, wavelength):
        self.d_x = d_x
        self.N_in = N_in
        self.z = z
        self.wavelength = wavelength
        self.wl_z = self.wavelength * self.z
        self.k = 2 * np.pi / self.wavelength
        self.max_freq = 1 / self.d_x

class FresnelSingle(Fresnel):
    def __init__(self, d_x, d_f, N_in, z, wavelength):
        super().__init__(d_x, N_in, z, wavelength)
        self.d_f = d_f
        self.ZP = self.calc_zero_padding()
        self.N_X = self.calc_phantom_length()

    def calc_phantom_length(self):
        """
        Calculate the equivalent zero-padded signal length N_X to achieve a specified output 
        frequency (d_f) for spatial sampling (d_x) using the single FT method

        Args: 
            d_x : Spatial sampling.
            d_f : Desired frequency sampling.
            N_in : Number of non-zero input samples.
            wavelength : Wavelength of the light.
            z: Propagation distance in m.

        Returns:
            float : N_X = Z_pad * N_in + 1 - phantom padded length of input signal
	    """
        N_X = self.max_freq * self.wl_z / self.d_f
        return N_X

    def calc_zero_padding(self):
        """
        Calculate the zero-padding factor (ZP * N_in) to achieve a specified frequency spacing
        (d_f) for a given spatial sampling (d_x) using the Fresnel single FT method.

        Args:
            d_x : Spatial sampling.
            d_f : Desired frequency spacing.
            N_in : Number of non-zero input samples.
            wavelength : Wavelength of the light.
            z: Propagation distance in m.

        Returns:
            float: Zero-padding factor (ZP * N_in) to achieve the desired frequency spacing.
        """
        ZP = ((self.max_freq * self.wl_z / self.d_f) - 1) / self.N_in
        return ZP

    def zoom_fresnel_single_fft(self, field, N_out):
        """
        Single FFT fresnel diffraction using Bluestein FFT

        Output on grid defined by (N_out/ZP*N_in)*wl_z/d_x, (N_out/ZP*N_in)*wl_z/d_x

        Args:
            field : 2D input field to be propagated. 
            d_x : Sampling interval of the input field [m]
            z : Propagation distance [m]
            wavelength : Wavelength of light [m]
            ZP : Zero-padding factor
            N_in :  Number of non-zero input samples in each dimension
            N_out : Number of output samples in each dimension

        Returns:
            tuple: The propagated output field and the output grid sampling
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

        Args
        x_file : Input mask as a memmap object
        N_out : Number of output samples.

        Returns
        output_field, df : Diffracted output field and sample size
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
        x_file = Input mask as a memmap object
        N_out : Number of output samples.

        Returns
        output_field, df : Diffracted output field and sample size
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
        # Use me if you'd rather not load a full-starshade or store one on memory
        # first call zoom_fft_quad_out_mod and compute the field on a single quadrant)
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
    Fresnel diffraction with a forwards-backwards Fourier transform 
    Reference: Introduction to Fourier Optics, Goodman. Ch 5. 
    Output spatial location same as input
    """
    def __init__(self, d_x, N_in, z, wavelength):
        super().__init__(d_x, N_in, z, wavelength)

    def fresnel_double_fft(self, field):
        """
        Double FFT fresnel diffraction

        Output on same grid as input

        Args:
            field : 2D input field to be propagated.
            d_x : Sampling interval of the input field [m]
            z : Propagation distance [m]
            wl : Wavelength of light [m]

        Returns:
            tuple: The propagated output field and the output grid sampling
        """
        field_copy = np.copy(field)
        k = 2 * np.pi / self.wavelength
        Ny, Nx = field_copy.shape
        df = self.max_freq / Nx
        freq_xy = grid_points(Nx, Ny, dx = df)
        spectral_field =  np.fft.fft2(field_copy) * np.exp(1j * self.z * k)\
         * np.exp(-1j * np.pi * self.wl_z * np.fft.fftshift(freq_xy[0]**2 + freq_xy[1]**2) ) 
        output_field = np.fft.ifft2(spectral_field)

        return output_field, df

class Fraunhofer:
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
        Calculate the equivalent zero-padded signal length N_X to achieve a specified output 
        frequency (d_f) for spatial sampling (d_x) using the single FT method

        Args: 
            d_x : Spatial sampling.
            d_f : Desired frequency sampling.
            N_in : Number of non-zero input samples.
            wavelength : Wavelength of the light.
            z: Propagation distance in m.

        Returns:
            float : N_X = Z_pad * N_in + 1 - phantom padded length of input signal
	    """
        N_X = self.max_freq * self.wl_z / self.d_f
        return N_X
        
    def calc_zero_padding(self):
        """
        Calculate the zero-padding factor (ZP * N_in) to achieve a specified frequency spacing
        (d_f) for a given spatial sampling (d_x) using the Fresnel single FT method.

        Args:
            d_x : Spatial sampling.
            d_f : Desired frequency spacing.
            N_in : Number of non-zero input samples.
            wavelength : Wavelength of the light.
            z: Propagation distance in m.

        Returns:
            float: Zero-padding factor (ZP * N_in) to achieve the desired frequency spacing.
        """
        ZP = ((self.max_freq * self.wl_z / self.d_f) - 1) / self.N_in
        return ZP

    def zoom_fraunhofer(self, field, N_out):
        """
        Fraunhofer diffraction at points within [ (N_out/ZP*N_in)*wl_z/d_x, (N_out/ZP*N_in)*wl_z/d_x ]
        using the Bluestein FFT.

        Args:
            field : 2D input field to be propagated. 
            d_x : Sampling interval of the input field [m]
            z : Propagation distance [m]
            wavelength : Wavelength of light [m]
            ZP : Zero-padding factor
            N_in :  Number of non-zero input samples in each dimension
            N_out : Number of output samples in each dimension

        Returns:
            tuple: The propagated output field and the output grid sampling
        """
        field_copy = np.copy(field)
        k = 2 * np.pi / self.wavelength
        Ny, Nx = field_copy.shape    
        output_field = zoom_fft_2d(field_copy, self.N_in, N_out, N_X = self.N_X) * (self.d_x**2)
        df = self.max_freq*self.wl_z / self.N_X
        out_xy = grid_points(N_out, N_out, dx = df )
        out_fac = np.exp ( ( 1j * k / (2 * self.z) ) * (out_xy[0]**2 + out_xy[1]**2) ) / (1j * self.wl_z)

        return out_fac*output_field, df
