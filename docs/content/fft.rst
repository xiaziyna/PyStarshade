Bluesteins FFT
---------------

When a starshade is aligned with a star, starlight is diffracted by the starshade :math:`s(\mathbf{x})` onto a telescope aperture. For a starshade mask :math:`s(u, v)` which is zero inside the mask and unity outside, at a wavelength :math:`\lambda` and starshade-telescope distance :math:`z`, the field at the telescope aperture is :math:`f_{\lambda}(x, y)` is related to the Fourier transform of the starshade mask multipled by a highly oscillatory chirp term:
.. math::
f_{\lambda}[u, v] \propto \mathcal{F} \left( s(u, v) e^{\frac{j \pi}{\lambda z} (u^2 + v^2)} \right) \left[ \frac{x}{\lambda z} ,\frac{y}{\lambda z} \right]

Numerical diffraction calculations must use a very small numerical resolution :math:`d u` of the starshade :math:`s(u, v)` in order to accurately calculate starlight suppression. Using a standard FFT to perform these calculations is inefficient as very large zero-padding factors are needed to sample the field at the telescope aperture. The Bluestein FFT is a technique to calculate arbitrary spectral samples of a propagated field, indirectly using FFTs and therefore benefiting from their efficiency. For an :math:`N \cdot N` starshade mask, and an :math:`M \cdot M` telescope aperture, the Bluestein FFT approach achieves a complexity of :math:`O((N+M)^2 \log (M+N))`. This technique is utilized in multiple aspects of the optical train to efficiently propagate fields.
