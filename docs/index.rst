.. PyStarshade documentation master file, created by
   sphinx-quickstart on Fri Dec  6 09:40:31 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyStarshade documentation
=========================

Developed by Jamila Taaki (MIDAS).

PyStarshade is a tool for simulating high-contrast imaging of exoplanets with starshades. 

What is a starshade? A starshade is a shaped mask flown in formation with a telescope to block starlight and image faint exoplanets.

Complex electric fields are calculated at three planes of propagation (starshade, telescope aperture and focal plane) using Fresnel or Fraunhofer diffraction formula where appropriate. 
When a starshade is aligned with a star, starlight is diffracted by the starshade $s(\mathbf{x})$ onto a telescope aperture. For a starshade mask $s(u, v)$ which is zero inside the mask and unity outside, at a wavelength $\lambda$ and starshade-telescope distance $z$, the field at the telescope aperture is $f_{\lambda}(x, y)$ is related to the Fourier transform of the starshade mask multipled by a highly oscillatory chirp term:
 $$f_{\lambda}[u, v] \propto \mathcal{F} \left( s(u, v) e^{\frac{j \pi}{\lambda z} (u^2 + v^2)} \right) \left[ \frac{x}{\lambda z} ,\frac{y}{\lambda z} \right]$$
Numerical diffraction calculations must use a very small numerical resolution $d u$ of the starshade $s(u, v)$ in order to accurately calculate starlight suppression. Using a standard FFT to perform these calculations is inefficient as very large zero-padding factors are needed to sample the field at the telescope aperture. The Bluestein FFT is a technique to calculate arbitrary spectral samples of a propagated field, indirectly using FFTs and therefore benefiting from their efficiency. For an $N \cdot N$ starshade mask, and an $M \cdot M$ telescope aperture, the Bluestein FFT approach achieves a complexity of $O((N+M)^2 \log (M+N))$. This technique is utilized in multiple aspects of the optical train to efficiently propagate fields.

`PyStarshade` allows for simulating imaging for a discretized exoplanetary scene of flux, varying starshade mask and telescope aperture mask (interfacing with HCIPy to generate mission telescope apertures). 

Usage
--------

Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.

Contents
--------

.. toctree::
    content/test
