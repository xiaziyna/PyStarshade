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

A key challenge in detecting exoplanets and characterizing their atmospheres, is distinguishing their faint reflected or emitted light from the star they orbit. Direct-imaging addresses this by suppressing starlight to reveal orbiting exoplanets [@direct]. Direct imaging can be achieved through two primary approaches: internal occulters, coronagraphs, which use optics within the telescope to block starlight, and external occulters, starshades, petal shaped masks some 10's of meters across flown in tandem with a telescope. Current direct imaging efforts, primarily using ground-based coronagraphs like the Very Large Telescope (VLT) and Gemini Observatory, have successfully imaged young, massive exoplanets at wide separations [@Follette_2023]. However, to see a faint Earth-like planet in the habitable zone of a Sun-like star will require extreme starlight suppression to the level of $10^{-8}$ - $10^{-10}$ [@angel]. 

While a starshade has not yet been flown in space, in subscale demonstrations a starshade suppresses starlight by a factor of $10^{-10}$ or more, across the broad wavelength range necessary for detecting water, oxygen, methane, and other biomarkers [@mennesson]. Beyond starlight suppression, critical questions remain regarding the properties and orbital configurations of exoplanetary systems observable by starshade missions, as well as optimal instrument designs. `PyStarshade` is a Python toolbox to efficiently perform optical simulations of starshades with exoplanetary scenes to assess the utility of starshades in future direct-imaging missions.

Several starshade concepts have been proposed but not adopted, including a 26 m diameter design to rendezvous with the Nancy Grace Roman Space Telescope  (a coronagraph scheduled for launch in the mid 2020's) [@ngrst_rendezvous]. A 52 m design as part of the Habex concept [@habex]. NASA’s Habitable Worlds Observatory (HWO), is in the early design phase with the goal to find and characterize a handful of earth-like exoplanets [@NAP26141]. A 60 m starshade operating over visible to infrared wavelengths (500–1000 nm), and a 35 m UV (250-500 nm) starshade, have been proposed for HWO [@hwo_ss].

# Statement of need
`PyStarshade` provides a toolbox for performing optical simulations from source to focal plane with a configurable starshade telescope design. Complex electric fields are calculated at three planes of propagation (starshade, telescope aperture and focal plane) using Fresnel or Fraunhofer diffraction formula where appropriate. These are used to pre-compute a spatially and spectrally varying point-spread-function (PSF) basis to simulate imaging. First-order imaging characteristics of a starshade can be determined from analytic relations which depend on the starshades size, telescope aperture size, wavelength and flight distance. `PyStarshade` allows for second order imaging characteristics to be studied, including simulations of imaging for a discretized exoplanetary scene of flux, varying starshade mask and telescope aperture mask to study throughput and post-processing methods. An example is provided on using spectral- and time-dependent scenes consisting of planets, star and dust-disk, generated with `ExoVista` [@Stark_2022], the output imaging is shown in \autoref{fig:example}. `PyStarshade` is intended to be flexible and efficient in studying i) exoplanet retrievals, and, ii) instrument design.

![Simulated imaging of a synthetic exoscene (ExoVista) with three visible exoplanets at a wavelength of 500 nm. A 60 m starshade configuration and a 6 m segmented pupil was used for this example. \label{fig:example}](exo_scene.png){ width=50% }

Fourier optics simulations of starshades and telescope apertures require fine spatial pixel scales to accurately model diffraction without aliasing errors. Traditional Fast Fourier Transforms (FFTs) necessitate large zero-padding factors to accomplish this, increasing computational cost. `PyStarshade` implements the Bluestein FFT [@bluestein] (related to the chirp Z-transform) to compute optical fields at arbitrary spatial scales, without zero-padding and therefore with greater efficiency than the FFT or discrete Fourier transform (DFT). In the associated paper submitted to AAS, we use `PyStarshade` to assess the complementary role of a 60 m HWO starshade [@hwo_ss] paired with 6 m segmented and obscured telescope apertures featuring cm-scale details. We report core throughput—the fraction of exoplanet light recovered—, a key metric governing exposure times needed to image an exoplanet and the overall yield of a mission.

Furthermore, `PyStarshade` implemens a novel technique to chunk Bluestein FFT calculations that mitigates memory bottlenecks. A set of unit tests are provided to validate the Fourier diffraction tools. `PyStarshade` pre-computes fields, point-spread-functions and throughput grids for different stages of the optical propagation chain, allowing for flexibility in modifying instrument parameters or telescope aperture masks. `PyStarshade` optionally interfaces with HCIPy [@hcipy] to generate telescope apertures and the resulting imaging. 

# Related software
The Starshade Imaging Simulation Toolkit for Exoplanet Reconnaissance (`SISTER`) [@sisters:2022] implemented in Matlab performs detailed end-to-end simulations of starshade imaging, further including instrument noise sources such as solar glint, and tools for modeling an exoplanetary system and background objects. The optical simulation pipeline of `PyStarshade` follows a similar approach to that of `SISTER`, employing a precomputed PSF basis for image simulation. However, `SISTER` computes diffraction from the starshade to the telescope aperture using the boundary diffraction wave method due to [@cady], performing a 1D integral along the starshade's edge, followed by a DFT to obtain the final PSF basis. Several codes focus on calculating the starshade diffraction incident on the telescope aperture; `Fresnaq` [@fresnaq] is a Matlab code to efficiently compute diffraction of a starshade via the non-uniform FFT [@barnett]. `Diffraq` [@diffraq] is a Python implementation of `Fresnaq`, this toolbox also provides an implementation of the boundary diffraction wave method and tools to generate starshade masks [@harness]. Starshade masks used in `PyStarshade` are obtained from `SISTER`, or were generated with `Diffraq`. Outside of imaging simulations, the Exoplanet Open-Source Imaging Mission Simulator (`EXOSIMS`) [@exosims] provides methods for orbital simulation of starshades and fuel consumption calculations.

# Acknowledgements

This project is supported by Schmidt Sciences, LLC.

# References
