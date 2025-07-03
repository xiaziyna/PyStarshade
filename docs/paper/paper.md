---
title: 'PyStarshade: simulating high-contrast imaging of exoplanets with starshades'
tags:
  - Python
  - astronomy
  - exoplanets
  - starshades
  - high-contrast imaging
authors:
  - name: Jamila Taaki
    orcid: 0000-0001-5475-1975
    affiliation: 1
    corresponding: true

  - name: Athol Kemball
    orcid: 0000-0001-6233-8347
    affiliation: "2, 3"

  - name: Farzad Kamalabadi
    affiliation: 2

affiliations:
 - name: Michigan Institute for Data Science, University of Michigan, USA
   index: 1
 - name: University of Illinois at Urbana-Champaign, USA
   index: 2
 - name: University of the Witwatersrand, Johannesburg, South Africa
   index: 3

date: 4 December 2024
bibliography: paper.bib
---

# Summary

A key challenge in detecting exoplanets and characterizing their atmospheres is distinguishing their faint reflected or emitted light from that of their host star. Direct imaging addresses this challenge by suppressing the star's light to reveal the orbiting exoplanet. There are two approaches to direct imaging: internal occulters, called coronagraphs, which use optics within the telescope to block starlight; and external occulters, called starshades, which are petal-shaped masks spanning tens of meters in diameter, flown in tandem with a telescope [@special]. Current direct imaging efforts using ground-based coronagraphs at the Very Large Telescope (VLT) and at the Gemini Observatory have successfully imaged young, massive exoplanets at wide separations [@Follette_2023]. However, observing a faint, Earth-like planet in the habitable zone of a Sun-like star requires extreme starlight suppression to the level of $10^{8}$ - $10^{10}$ star-to-planet flux ratio [@angel].

Although no starshade has been flown in space yet, subscale starshades have demonstrated $\leq 10^{9}$ starlight suppression at a wavelength of 633 nm in laboratory demonstrations [@subscale], and $\leq 10^{10}$ across the broad wavelength range necessary for detecting biomarkers such as water, oxygen, and methane in computational simulations [@mennesson]. Beyond demonstrating starlight suppression, critical questions remain regarding the properties and orbital configurations of exoplanetary systems that can be observed by starshade missions, as well as optimal instrument designs. `PyStarshade` is a Python toolbox that efficiently performs optical simulations of starshades with exoplanetary scenes to assess the utility of starshades in future direct-imaging missions.

Several starshade concepts have been proposed but not yet adopted. These include: a 26-meter-diameter design to rendezvous with the Nancy Grace Roman Space Telescope (scheduled for launch in 2026) [@ngrst_rendezvous]; a 52-meter design as part of the Habex concept [@habex];  a 60-meter starshade operating over visible-to-infrared wavelengths 500–1,000 nm) and a 35-meter UV (250–500 nm) starshade, proposed for the Habitable Worlds Observatory (HWO), a NASA mission is in the early design phase, aiming to find and characterize a handful of Earth-like exoplanets [@NAP26141]. An example use of  `PyStarshade` to evaluate the complementary role of a 60-meter HWO starshade [@hwo_ss] paired with two different 6-meter segmented and obscured telescope apertures featuring centimeter-scale details is presented in [@taaki_hwo_sim_2025] where we report $core$ $throughput$ for these apertures. Core throughput is defined as the fraction of exoplanet light recovered and is a key metric that governs the exposure time needed to image an exoplanet and therefore the overall exoplanetary yield of a mission.

# Statement of need
`PyStarshade` provides a toolbox for performing optical simulations from source to focal plane with a configurable starshade telescope design. Complex electric fields are calculated at three planes of propagation: the starshade, the telescope aperture, and the focal plane, using the Fresnel or Fraunhofer diffraction formula where appropriate. First-order  imaging characteristics of a starshade can be determined from analytic relations that depend on the size of the starshade, the size of the telescope aperture, the wavelength of the observations, and the flight distance. `PyStarshade` allows for second-order imaging characteristics to be studied, including imaging simulations with a pixelized exoplanetary input scene of integrated flux per pixel and wavelength. It also allows for the study of throughput and post-processing methods with varying starshade masks and telescope aperture masks to study throughput and post-processing methods. An example imaged scene is shown in \autoref{fig:example} at a snapshot in time and wavelength. The scene consists of a planet, a star, and a dust disk. These spectral- and time-dependent scenes were generated with ExoVista [@Stark_2022]. `PyStarshade` is intended to be flexible and efficient in studying exoplanet retrievals and instrument design.

![A starshade imaging simulation shown at a wavelength of 500 nm with a synthetic exoplanetary input scene (generated with ExoVista): three exoplanets are directly visible, while two more sit inside the starshade suppression zone. The scene assumes a 60 m HWO starshade paired with a 6 m segmented telescope; the planets in the scene have planet-to-star flux ratios between $10^{-8}$ and $10^{-10}$. \label{fig:example}](exo_scene.png){ width=50% }

Fourier optics simulations of starshades and telescope apertures require fine spatial pixel scales to accurately model diffraction without aliasing errors across multiple stages of propagation. Traditional fast Fourier transforms (FFTs) require large zero-padding factors to accomplish this, increasing computational cost. `PyStarshade` implements the Bluestein FFT [@bluestein] (related to the chirp Z-transform) to compute optical fields at arbitrary spatial scales without zero-padding, thereby achieving greater efficiency than the FFTs or discrete Fourier transforms (DFTs). `PyStarshade`implements the Bluestein FFT based on `Numpy` [@numpy] FFTs.

Furthermore, `PyStarshade` implements a novel technique to chunk Bluestein FFT calculations, which mitigates memory bottlenecks. A set of unit tests is provided to validate the Fourier diffraction tools. `PyStarshade` precomputes fields, point spread functions, and throughput grids for different stages of the optical propagation chain. This allows flexibility in modifying instrument parameters or telescope aperture masks. `PyStarshade` optionally interfaces with HCIPy [@hcipy] to generate telescope apertures and the resulting imaging.

# Related software
The Starshade Imaging Simulation Toolkit for Exoplanet Reconnaissance (`SISTER`) [@sisters:2022], implemented in MATLAB, performs detailed, end-to-end simulations of starshade imaging, further including instrument noise sources such as solar glint, and tools for modeling an exoplanetary system and background objects. The optical simulation pipeline of `PyStarshade` follows a similar approach to that of `SISTER`, employing a precomputed PSF basis for image simulation. However, SISTER computes diffraction from the starshade to the telescope aperture using the boundary diffraction wave method [@cady], performing a 1D integral along the starshade's edge, followed by a DFT to obtain the final PSF basis. Several other libraries focus exclusively on calculating the starshade diffraction incident on the telescope aperture. One such library is `fresnaq`, a MATLAB library that efficiently computes diffraction by a starshade via the non-uniform FFT [@barnett]. `diffraq` [@diffraq] is a Python implementation of `freqnaq`. This library also provides an implementation of the boundary diffraction wave method and tools to generate starshade masks [@harness]. The starshade masks used in `PyStarshade` are obtained from `SISTER` or were generated with `diffraq`. Outside of imaging simulations, the Exoplanet Open-Source Imaging Mission Simulator (`EXOSIMS`) [@exosims] provides methods for orbital simulation of starshades and fuel consumption calculations.

# Acknowledgements

This project is supported by Schmidt Sciences, LLC.

# References
