import numpy as np

from pystarshade.diffraction.bluestein_fft import zoom_fft_2d_mod
from pystarshade.diffraction.util import *


class SourceField:
    """
    Computes far-field diffraction for an N_s x N_s source field using a Bluestein FFT.

    Attributes
    ----------
    d_s : float
        Source field sampling.
    N_s : int
        Number of samples in the field.
    wavelength : float
        Wavelength of light.
    z : float
        Far-field distance.
    source_field : np.ndarray
        The source field array.

    Notes
    -----
    The class uses a plane wave description of the far-field. Either `d_s_mas` or `d_s` and `z` must be defined.
    """

    def __init__(self, d_s, N_s, wavelength, z, source_field):
        self.N_s = N_s
        self.wavelength = wavelength
        self.d_s = d_s
        self.z = z
        self.k = 2 * np.pi / self.wavelength
        self.max_freq = 1 / self.d_s
        self.source_field = source_field

    def farfield(self, d_x, N_x, z1):
        """
        Calculate the far-field diffraction of a source field at a local distance `z1` over an (N_x, N_x) grid.
        Note: z1 is defined as a local distance, the origin is set in the far-field.

        Parameters
        ----------
        d_x : float
            Output spatial sampling.
        N_x : int
            Number of output samples.
        z1 : float
            Propagation distance (Note: computationally, the origin is treated as the starshade plane).

        Returns
        -------
        np.ndarray
            A 2D array representing the computed field over the grid.
        """
        ZP = (
            (self.max_freq * self.wavelength * (self.z) / d_x) - 1
        ) / self.N_s
        out_field = zoom_fft_2d_mod(self.source_field, self.N_s, N_x, Z_pad=ZP)
        out_fac = np.exp(1j * self.k * z1)
        return out_fac * out_field


class Field:
    """
    This class returns analytic far-field diffraction for sources, or can be used to generate source fields

    Attributes
    ----------
    d_x : float
        Spatial sampling of the field.
    N : int
        Number of samples in the field.
    wavelength : float
        Wavelength of light.
    z : float
        Far-field distance.

    Notes
    -----
    Includes plane-wave fields, Gaussian sources, and point sources.
    """

    def __init__(self, d_x, N, wavelength, z):
        self.d_x = d_x
        self.N = N
        self.wavelength = wavelength
        self.z = z

    def update(self, d_x=None, N=None):
        """
        Update the sampling (`d_x`) and number of samples (`N`).

        Parameters
        ----------
        d_x : float, optional
            New value for spatial sampling.
        N : int, optional
            New value for the number of samples.
        """
        if d_x is not None:
            self.d_x = d_x
        if N is not None:
            self.N = N


class GaussianSource(Field):
    """
    Computes far-field diffraction for a Gaussian source.

    Attributes
    ----------
    d_x : float
        Spatial sampling of the output field.
    N : int
        Number of samples in the output field.
    wavelength : float
        Wavelength of light.
    x : float
        x-coordinate of the Gaussian source origin.
    y : float
        y-coordinate of the Gaussian source origin.
    z : float
        Far-field distance.
    A : float
        Amplitude of the Gaussian source.
    sigma : float
        Standard deviation of the Gaussian source.

    Inherits From
    -------------
    Field

    Notes
    -----
    The source field is given by `A * exp(-(x/sigma)^2)`.
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
        Calculate far-field amplitude (`A`) and inverse standard deviation (`1/sigma`).

        Returns
        -------
        tuple of float
            - `far_A` : Far-field amplitude.
            - `far_inv_sigma` : Inverse standard deviation of the far-field Gaussian.

        Notes
        -----
        See: https://www.pas.rochester.edu/~dmw/ast203/Lectures/Lect_14.pdf
        """
        far_A = self.A * np.pi * self.sigma**2 / self.wl_z
        far_inv_sigma = np.pi * self.sigma / self.wl_z
        return far_A, far_inv_sigma

    def far_field_gaussian(self, far_A, far_inv_sigma, z1):
        """
        Compute the far-field Gaussian on a Cartesian grid.

        Parameters
        ----------
        far_A : float
            Far-field amplitude.
        far_inv_sigma : float
            Inverse standard deviation of the far-field Gaussian.
        z1 : float
            Propagation distance (Note: the origin is treated as the starshade plane).

        Returns
        -------
        np.ndarray
            A 2D array representing the far-field Gaussian on the grid.
        """
        vals = np.linspace(-(self.N // 2), (self.N // 2), self.N) * self.d_x
        xx, yy = np.meshgrid(vals, vals)

        return (
            far_A
            * np.exp(-(xx**2 + yy**2) * far_inv_sigma**2)
            * np.exp(1j * 2 * np.pi * z1 / self.wavelength)
        )


class PointSource(Field):
    """
    Computes far-field diffraction for a point source.

    Attributes
    ----------
    d_x : float
        Spatial sampling of the output field.
    N : int
        Number of samples in the output field.
    wavelength : float
        Wavelength of light.
    x : float
        x-coordinate of the source.
    y : float
        y-coordinate of the source.
    z : float
        Far-field distance.
    A : float
        Amplitude of the source.

    Inherits From
    -------------
    Field
    """

    def __init__(self, d_x, N, wavelength, x, y, z, A):
        super().__init__(d_x, N, wavelength, z)
        self.x = x
        self.y = y
        self.A = A

    def wave_numbers(self):
        """
        Calculate planar wave numbers for a point source at position `(x, y, z)`.

        Returns
        -------
        np.ndarray
            A 1D array `[k_x, k_y, k_z]` representing planar wave numbers.

        Notes
        -----
        Based on Blahut (Theory of Remote Image Formation, sec 1.7).
        """
        if self.x == 0 and self.y == 0:
            theta = 0
        else:
            theta = np.hypot(self.x, self.y) / self.z
        if self.x == 0:
            phi = np.pi / 2
        else:
            phi = np.arccos(self.x / np.hypot(self.x, self.y))
        ax = np.cos(phi) * np.sin(theta)
        by = np.sin(theta) * np.sin(phi)
        cz = np.cos(theta)
        k = np.array([ax, by, cz]) * 2 * np.pi / self.wavelength
        return k

    def plane_wave(self, k, z1):
        """
        Compute a plane wave with amplitude `A` and wavevector `k` on a Cartesian grid.

        Parameters
        ----------
        k : np.ndarray
            Wave vector `[k_x, k_y, k_z]`.
        z1 : float
            Propagation distance (Note: computationally, the origin is treated as the starshade plane).

        Returns
        -------
        np.ndarray
            A 2D array representing the computed plane wave on the grid.
        """
        vals = np.linspace(-(self.N // 2), (self.N // 2), self.N) * self.d_x
        xx, yy = np.meshgrid(vals, vals)
        vec = np.array([xx, yy]).T
        return (
            self.A * np.exp(1j * np.inner(k[:2], vec)) * np.exp(1j * k[2] * z1)
        )
