import numpy as np
from diffraction.bluestein_fft import zoom_fft_2d_mod, zoom_fft_2d
from diffraction.util import *

class Fresnel:
    def __init__(self, d_x, N_in, z, wavelength):
        self.d_x = d_x
        self.N_in = N_in
        self.z = z
        self.wavelength = wavelength
        self.wl_z = self.wavelength * self.z
        self.max_freq = 1 / self.d_x

class FresnelSingle(Fresnel):
    def __init__(self, d_x, d_f, N_in, z, wavelength):
        super().__init__(d_x, N_in, z, wavelength)
        self.d_f = d_f
        self.ZP = self.calc_zero_padding()

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
        ZP = np.ceil(ZP)
        ZP += ZP % 2
        return ZP

    def zoom_fresnel_single_fft(self, field, N_out):
        """
        Single FFT fresnel diffraction using Bluestein FFT

        Output on grid defined by (N_out/ZP*N_in)*wl_z/d_x, (N_out/ZP*N_in)*wl_z/d_x

        Args:
            field : 2D input field to be propagated. Field should be at least the size of (N_in + N_out - 1)
            d_x : Sampling interval of the input field [m]
            z : Propagation distance [m]
            wavelength : Wavelength of light [m]
            ZP : Zero-padding factor
            N_in :  Number of non-zero input samples in each dimension
            N_out : Number of output samples in each dimension

        Returns:
            tuple: The propagated output field and the output grid sampling
        """
        k = 2 * np.pi / self.wavelength
        Ny, Nx = field.shape
        in_xy = grid_points(Nx, Ny, dx = self.d_x)

        output_field = zoom_fft_2d_mod(field * np.exp(1j * (np.pi /self.wl_z) * (in_xy[0]**2 + in_xy[1]**2)), self.N_in, N_out, self.ZP) * (self.d_x**2)
        df = (self.max_freq*self.wl_z / (self.ZP*self.N_in + 1))
        out_xy = grid_points(N_out, N_out, dx = df )

        quad_out_fac = np.exp(1j * k * self.z) * np.exp(1j * k / (2 * self.z) * (out_xy[0]**2 + out_xy[1]**2)) / ( 1j * self.wl_z) 
        return quad_out_fac * output_field, df

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
        k = 2 * np.pi / self.wavelength
        Ny, Nx = field.shape
        df = self.max_freq / Nx
        freq_xy = grid_points(Nx, Ny, dx = df)
        spectral_field = np.fft.fft2(field) * np.exp(1j * self.z * k) * np.exp(-1j * np.pi * self.wl_z * np.fft.fftshift(freq_xy[0]**2 + freq_xy[1]**2) ) 
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

    def calc_zero_padding(self):
        """
        Calculate approximate zero-padding factor (ZP * N_in) to achieve a specified frequency spacing
        (d_f) for a given spatial sampling (d_x) using the Fraunhofer prop.

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
        ZP = np.ceil(ZP)
        ZP += ZP % 2
        return ZP

    def zoom_fraunhofer(self, field, N_out):
        """
        Fraunhofer diffraction at points within [ (N_out/ZP*N_in)*wl_z/d_x, (N_out/ZP*N_in)*wl_z/d_x ]
        using the Bluestein FFT.

        Args:
            field : 2D input field to be propagated. Field should be at least the size of (N_in + N_out - 1)
            d_x : Sampling interval of the input field [m]
            z : Propagation distance [m]
            wavelength : Wavelength of light [m]
            ZP : Zero-padding factor
            N_in :  Number of non-zero input samples in each dimension
            N_out : Number of output samples in each dimension

        Returns:
            tuple: The propagated output field and the output grid sampling
        """
        k = 2 * np.pi / self.wavelength
        Ny, Nx = field.shape    
        output_field = zoom_fft_2d(field, self.N_in, N_out, self.ZP) * (self.d_x**2)
        df = self.max_freq*self.wl_z / (self.ZP*self.N_in + 1)
        out_xy = grid_points(N_out, N_out, dx = df )
        out_fac = np.exp ( ( 1j * k / (2 * self.z) ) * (out_xy[0]**2 + out_xy[1]**2) ) / (1j * self.wl_z)

        return out_fac*output_field, df
