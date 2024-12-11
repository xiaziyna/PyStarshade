.. autoclass:: pystarshade.propagator.StarshadeProp
   :members: set_mission_params, gen_pupil_field, gen_pupil, gen_psf_basis, gen_scene


.. autoclass:: pystarshade.diffraction.diffract.Fresnel
    :members:

.. autoclass:: pystarshade.diffraction.diffract.FresnelSingle
    :members:  calc_phantom_length, calc_zero_padding, zoom_fresnel_single_fft, nchunk_zoom_fresnel_single_fft

.. autoclass:: pystarshade.diffraction.diffract.Fraunhofer
    :members: calc_phantom_length, calc_zero_padding, zoom_fraunhofer

.. autoclass:: pystarshade.diffraction.field.SourceField
    :members: farfield

.. autoclass:: pystarshade.diffraction.field.Field
    :members: update

.. autoclass:: pystarshade.diffraction.field.PointSource
    :members: wave_numbers, plane_wave
