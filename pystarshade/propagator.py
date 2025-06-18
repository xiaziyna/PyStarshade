from pystarshade.data.drm import telescope_params, mas_to_rad
import glob
import numpy as np
from pystarshade.simulate_field import source_field_to_pupil, pupil_to_ccd
from pystarshade.apodization.pupil import make_pupil
from pystarshade.diffraction.util import flat_grid, bluestein_pad, trunc_2d, pad_2d, data_file_path, h5_to_npz
import h5py

class StarshadeProp:
    """
    Pre-compute files and propagate light past a starshade.

    Parameters
    ----------
    drm : str
        Design reference mission (e.g., 'wfirst' or 'hwo').
    d_s_mas : float, optional
        Source angular resolution in milliarcseconds, defaults to 2.
    d_t_mas : float, optional
        Pixel shift in the pupil plane corresponding to the angular field, defaults to 0.05.
    d_p_mas : float, optional
        Pixel size in the focal plane in milliarcseconds, defaults to 2.
    d_wl : float, optional
        Wavelength step size in nanometers, defaults to 50.

    Examples
    --------
    >>> hwo_starshade = StarshadeProp(drm='hwo')
    >>> hwo_starshade.gen_pupil_field()
    >>> hwo_starshade.gen_psf_basis(pupil_type='hex')
    >>> hwo_starshade.gen_scene(pupil_type='hex', source_field, 500e-9)
    """
    def __init__(self, drm = None, d_s_mas = 2, d_t_mas = 0.05, d_p_mas = 2, d_wl = 50):
        self.drm = drm
        self.d_s_mas = d_s_mas
        self.d_t_mas = d_t_mas # each pixel shift in the pupil plane corresponds to the field from an d_t_mas off-axis point source
        self.ratio_s_t = int(self.d_s_mas // self.d_t_mas)
        self.d_p_mas = d_p_mas # each pixel in the focal plane corresponds to d_p_mas of the source field
        self.ratio_s_p = int(self.d_s_mas // self.d_p_mas)
        self.d_wl = d_wl

        if not self.drm:
            raise ValueError("Design reference mission not set. Insert instrument paramters into data/telescope_drm.py")
        else: self.set_mission_params(self.drm)

    def set_mission_params(self, drm, mask_choice = 1, band_i = 0):
        """
        Load the design reference mission (defined inside `data/telescope_drm.py`).

        Parameters
        ----------
        drm : str
            Design reference mission ('wfirst', 'hwo', etc.).
        mask_choice : int, optional
            Mask sampling choice.
        band_i : int, optional
            Wavelength band index.
        """
        drm_params = telescope_params[drm]
        self.f = drm_params['focal_length_lens']
        self.suppress_region = int(2 * drm_params['iwa'][band_i])
        self.suppress_region += self.suppress_region%2
        self.r_lens = drm_params['radius_lens']
        self.mask_choice = mask_choice

        self.N_x =  int(2*(drm_params['ss_radius']+.2)*10 / (drm_params['dx_'][mask_choice]*10)) + 1
        self.d_x = drm_params['dx_'][mask_choice]
        self.d_x_str = drm_params['grey_mask_dx'][mask_choice]
        self.ss_mask_fname_partial =  data_file_path('grey_%s_%s_mask_%s' % (drm, str(drm_params['num_pet']), self.d_x_str), 'masks', 'starshade_masks')

        # wavelength bands
        wl_min, wl_max = drm_params['wl_bands'][band_i][0], drm_params['wl_bands'][band_i][1]
        self.iwa = drm_params['iwa'][band_i]
        self.dist_ss_t = drm_params['dist_ss_t'][band_i]

        self.wl_range = np.arange(wl_min, wl_max, self.d_wl, dtype=np.float128)*1e-9
        self.N_wl = int(len(self.wl_range))

        self.ang_res_mas = (self.wl_range /(2*self.r_lens))/mas_to_rad
        self.ang_res_pix = self.ang_res_mas/self.d_p_mas

        # set telescope sampling
        self.d_t = self.d_t_mas * self.dist_ss_t * mas_to_rad
        self.N_t = int((self.r_lens * 2) // self.d_t)
        self.N_t += (1 - self.N_t%2)
        self.d_p = (self.d_p_mas) * self.f * mas_to_rad # physical pixel size
        
    def mirr_symm_psf(self, psf_basis, N_basis, N_pix):
        """
        Mirror a quadrant of a symmetric psf object four ways.

        Parameters
        ----------
        psf_basis : np.ndarray
            quadrant of a psf basis
        N_basis : int
            full-size outpu5 number of PSF's along one axis
        N_pix : int
            number of pixels along one axis for a PSF

        Returns
        -------
        np.ndarray
            Mirrored PSF basis
        """

        psf_basis_mirrored = np.zeros((N_basis, N_basis, N_pix, N_pix), dtype=np.float32)
        N_basis_half = N_basis//2      
        psf_basis_mirrored[:N_basis_half, N_basis_half+1:] = np.flipud(psf_basis[1:, 1:])
        psf_basis_mirrored[N_basis_half:, N_basis_half:] = psf_basis
        psf_basis_mirrored[N_basis_half+1:, :N_basis_half] = np.fliplr(psf_basis[1:, 1:])
        psf_basis_mirrored[:N_basis_half+1, :N_basis_half+1] = np.fliplr(np.flipud(psf_basis))
        return psf_basis_mirrored

    def calc_magnification(self, dist_xo_ss):
        """
        Calculate the magnification of the field-of-view for a telescope focal number.

        Parameters
        ----------
        dist_xo_ss : float
            Distance from the observer to the starshade.

        Returns
        -------
        float
            Magnification factor.
        """
        return self.f / (dist_xo_ss + self.dist_ss_t)

    def calc_d_s(self, d_s_mas, dist_xo_ss):
        """
        Calculate the sampling size in meters.

        Parameters
        ----------
        d_s_mas : float
            Sampling size in milliarcseconds.
        dist_xo_ss : float
            Distance from observer to starshade.

        Returns
        -------
        float
            Sampling size in meters.
        """
        return (d_s_mas * (dist_xo_ss * mas_to_rad))

    def calc_d_s_mas(self, d_s, dist_xo_ss):
        return (d_s / dist_xo_ss) / mas_to_rad

    def gen_pupil_field(self, chunk = 1):
        """
        Generate the field at the pupil for the chosen starshade.

        Parameters
        ----------
        chunk : int, optional
            Whether to use chunked parallel processing (if so, must use a memmap file).
        """

        fname = data_file_path(f"{self.drm}_pupil_{self.d_x_str}*.npz", 'fields')
        if glob.glob(fname): return

        print("Pupil field does not exist. Generating.")
        if chunk:
            ss_mask_fname = self.ss_mask_fname_partial + '.dat'
        else:
            ss_mask_fname = self.ss_mask_fname_partial + '_qu.npz'

        over_N_t = 2*int((self.suppress_region // self.d_t_mas) + 100)
        over_N_t += (1-over_N_t%2)
        for wl_i in self.wl_range:
            save_path = data_file_path(self.drm+'_pupil_'+self.d_x_str+'_'+str(int(wl_i * 1e9))+'.npz', 'fields')
            field_incident_telescope, field_free_prop, params = source_field_to_pupil(ss_mask_fname, wl_i,\
            self.dist_ss_t, N_x = self.N_x, N_t = over_N_t, dx = self.d_x, dt = self.d_t, chunk=chunk)
            np.savez_compressed(save_path, field=field_incident_telescope, freesp_field=field_free_prop, params=params)

    def gen_pupil(self, pupil_type):
        """
        Generate a pupil mask (uses HCIPy).
        Pupil options are:
            circ, ELT, GMT, TMT, Hale, LUVOIR-A, LUVOIR-B, Magellan, VLT, HiCAT, HabEx, HST, JWST, Keck, hex

        Parameters
        ----------
        pupil_type : str
            The type of pupil aperture (e.g., 'circ', 'hex').
        """
        file_path = data_file_path(pupil_type+'_'+str(int(self.N_t))+'.npz', 'pupils')
        try:
            pupil_data = np.load(file_path)
            self.pupil_mask = pupil_data['pupil']
        except FileNotFoundError:
            print(f"Generating a new pupil mask.")
            make_pupil(self.N_t, pupil_type, file_path)
            pupil_data = np.load(file_path)
            self.pupil_mask = pupil_data['pupil']

    def gen_psf_basis(self, pupil_type, pupil_symmetry = False):
        """
        Generate the incoherent PSF basis for a particular pupil and starshade.

        Contrast is measured in units of if no starshade were present, i.e., just FFT of the pupil_mask.
        If the pupil is symmetric, only generate the positive quadrant of PSFs.

        Parameters
        ----------
        pupil_type : str
            The type of pupil aperture (e.g., 'circ', 'hex').
        pupil_symmetry : bool, optional
            Whether to use symmetry to reduce computation, defaults to False.
        """
        fname = data_file_path(f"{self.drm}_psf_{pupil_type}_{self.d_x_str}*.npz", 'psf')
        if glob.glob(fname): return

        print("PSF file does not exist. Generating: ")

        self.gen_pupil(pupil_type)
        N_basis = (2-pupil_symmetry)*(self.suppress_region // self.d_s_mas)
        N_basis += 1 - N_basis%2
        N_pix = int(( 20 * self.wl_range[-1] / (mas_to_rad*2*self.r_lens) ) // self.d_p_mas)
        N_pix += 1 - N_pix%2
        N_pix_overcomplete = int(( 160 * self.wl_range[-1] / (mas_to_rad*2*self.r_lens) ) // self.d_p_mas)
        N_pix_overcomplete += 1-N_pix_overcomplete%2
        x, y = np.meshgrid(np.arange(-(N_pix // 2), (N_pix // 2) + 1), np.arange(-(N_pix // 2), (N_pix // 2) + 1))
        core_throughput = np.zeros((self.N_wl, N_basis, N_basis))
        total_throughput = np.zeros((self.N_wl, N_basis, N_basis))

        psf_points = flat_grid(N_basis, negative = 1 - pupil_symmetry)
        for wl_i in range(self.N_wl):
            data = np.load(data_file_path(self.drm+'_pupil'+'_'+self.d_x_str+'_'+str(int(self.wl_range[wl_i]*1e9))+'.npz', 'fields'))
            wl = data['params'][0]
            save_path_h5 = data_file_path(self.drm+'_psf_'+pupil_type+'_'+self.d_x_str+'_'+str(int(wl * 1e9))+'.h5','psf')
            save_path_npz = data_file_path(self.drm+'_psf_'+pupil_type+'_'+self.d_x_str+'_'+str(int(wl * 1e9))+'.npz','psf')

            pupil_field = data['field'] 
            over_N_t = int(np.shape(pupil_field)[0])
            pupil_field_no_ss = trunc_2d(data['freesp_field'], self.N_t)

            focal_field_no_ss = pupil_to_ccd(wl, self.f, pupil_field_no_ss, self.pupil_mask, self.d_t, self.d_p, self.N_t, N_pix_overcomplete)
            norm_contrast = np.max(np.abs(focal_field_no_ss)**2)
            norm_factor = np.sum(np.abs(focal_field_no_ss)**2)
            circ_mask = np.hypot(x, y) <= self.ang_res_pix[wl_i] * 0.7
            with h5py.File(save_path_h5, 'w') as f:
                psf_ds = f.create_dataset('psf_basis', shape=(N_basis, N_basis, N_pix, N_pix), dtype='float32', compression='lzf', chunks=(1, 1, N_pix, N_pix)) 
                on_axis_ds = f.create_dataset('on_axis_psf', shape=(N_pix_overcomplete, N_pix_overcomplete), dtype='float64', compression='lzf')
                no_ss_ds = f.create_dataset('no_ss_psf', shape=(N_pix_overcomplete, N_pix_overcomplete), dtype='float32',compression='lzf')
                params_ds = f.create_dataset('params', shape = (7,), dtype='float32')
                no_ss_ds[:] = np.abs(focal_field_no_ss)**2 / norm_factor
                del focal_field_no_ss

                for (i, j) in psf_points:
                    if pupil_symmetry: p_i, p_j = i, j
                    else: p_i, p_j = i + N_basis//2, j + N_basis//2
                    if i == 0 and j == 0:
                        on_axis_field = trunc_2d(pupil_field, self.N_t)
                        focal_field = pupil_to_ccd(wl, self.f, on_axis_field, self.pupil_mask, self.d_t, self.d_p, self.N_t, N_pix_overcomplete)
                        on_axis_psf = np.abs(focal_field)**2 
                        psf_ = trunc_2d(on_axis_psf, N_pix)
                        core_throughput[wl_i, p_i, p_j] = np.sum(circ_mask*psf_) / norm_factor
                        total_throughput[wl_i, p_i, p_j] = np.sum(on_axis_psf) / norm_factor
                        on_axis_ds[:] = on_axis_psf / norm_factor
                        del on_axis_field, focal_field, on_axis_psf
                    else:
                        off_axis_field = pupil_field[(over_N_t//2) - (self.N_t//2) + i*self.ratio_s_t: (over_N_t//2) + (self.N_t//2) + 1 + i*self.ratio_s_t, \
                                                    (over_N_t//2) - (self.N_t//2) + j*self.ratio_s_t: (over_N_t//2) + (self.N_t//2) + 1 + j*self.ratio_s_t ]
                        focal_field = pupil_to_ccd(wl, self.f, off_axis_field, self.pupil_mask, self.d_t, self.d_p, self.N_t, N_pix)
                        psf_ = np.abs(focal_field).astype(np.float32)**2 
                        core_throughput[wl_i, p_i, p_j] = np.sum(circ_mask*psf_) / norm_factor
                        total_throughput[wl_i, p_i, p_j] = np.sum(psf_) / norm_factor
                        psf_ds[p_i, p_j, :, :] = psf_ / norm_factor

                params = np.array([ wl, self.d_p_mas, norm_contrast, norm_factor, N_basis, N_pix, N_pix_overcomplete])
                params_ds[:] = params
                h5_to_npz(save_path_h5, save_path_npz)
        
        save_path_throughput = data_file_path(self.drm+'_throughput_'+pupil_type+'_'+self.d_x_str+'.npz','psf')
        np.savez_compressed(save_path_throughput, core_throughput=core_throughput, total_throughput=total_throughput,\
                            grid_points = psf_points, wl=self.wl_range, d_pix_mas = self.d_p_mas)


    def gen_psf_basis_(self, pupil_type, pupil_symmetry = False):
        """
        Generate the incoherent PSF basis for a particular pupil and starshade.

        Contrast is measured in units of if no starshade were present, i.e., just FFT of the pupil_mask.
        If the pupil is symmetric, only generate the positive quadrant of PSFs.

        Parameters
        ----------
        pupil_type : str
            The type of pupil aperture (e.g., 'circ', 'hex').
        pupil_symmetry : bool, optional
            Whether to use symmetry to reduce computation, defaults to False.
        """
        fname = data_file_path(f"{self.drm}_psf_{pupil_type}_{self.d_x_str}*.npz", 'psf')
        if glob.glob(fname): return

        print("PSF file does not exist. Generating: ")

        self.gen_pupil(pupil_type)
        N_basis = (2-pupil_symmetry)*(self.suppress_region // self.d_s_mas)
        N_basis += 1 - N_basis%2
        N_pix = int(( 20 * self.wl_range[-1] / (mas_to_rad*2*self.r_lens) ) // self.d_p_mas)
        N_pix += 1 - N_pix%2
        N_pix_overcomplete = int(( 160 * self.wl_range[-1] / (mas_to_rad*2*self.r_lens) ) // self.d_p_mas)
        N_pix_overcomplete += 1-N_pix_overcomplete%2
        x, y = np.meshgrid(np.arange(-(N_pix // 2), (N_pix // 2) + 1), np.arange(-(N_pix // 2), (N_pix // 2) + 1))
        core_throughput = np.zeros((self.N_wl, N_basis, N_basis))
        total_throughput = np.zeros((self.N_wl, N_basis, N_basis))

        psf_points = flat_grid(N_basis, negative = 1 - pupil_symmetry)

        for wl_i in range(self.N_wl):
            psf_basis = np.zeros((N_basis, N_basis, N_pix, N_pix), dtype=np.float32)
            on_axis_psf = np.zeros((N_pix_overcomplete, N_pix_overcomplete), dtype=np.float64)
            data = np.load(data_file_path(self.drm+'_pupil'+'_'+self.d_x_str+'_'+str(int(self.wl_range[wl_i]*1e9))+'.npz', 'fields'))
            wl = data['params'][0]
            pupil_field = data['field']
            over_N_t = int(np.shape(pupil_field)[0])
            pupil_field_no_ss = trunc_2d(data['freesp_field'], self.N_t)
            focal_field_no_ss = pupil_to_ccd(wl, self.f, pupil_field_no_ss, self.pupil_mask, self.d_t, self.d_p, self.N_t, N_pix_overcomplete)
            norm_contrast = np.max(np.abs(focal_field_no_ss)**2)
            norm_factor = np.sum(np.abs(focal_field_no_ss)**2)
            circ_mask = np.hypot(x, y) <= self.ang_res_pix[wl_i] * 0.7

            for (i, j) in psf_points:
                if pupil_symmetry: p_i, p_j = i, j
                else: p_i, p_j = i + N_basis//2, j + N_basis//2

                if i == 0 and j == 0:
                    on_axis_field = trunc_2d(pupil_field, self.N_t)
                    focal_field = pupil_to_ccd(wl, self.f, on_axis_field, self.pupil_mask, self.d_t, self.d_p, self.N_t, N_pix_overcomplete)
                    on_axis_psf = np.abs(focal_field)**2 #/ norm_factor
                    psf_basis[p_i, p_j] = trunc_2d(on_axis_psf, N_pix) #/ norm_factor
                else:
                    off_axis_field = pupil_field[(over_N_t//2) - (self.N_t//2) + i*self.ratio_s_t: (over_N_t//2) + (self.N_t//2) + 1 + i*self.ratio_s_t, \
                                                 (over_N_t//2) - (self.N_t//2) + j*self.ratio_s_t: (over_N_t//2) + (self.N_t//2) + 1 + j*self.ratio_s_t ]
                    focal_field = pupil_to_ccd(wl, self.f, off_axis_field, self.pupil_mask, self.d_t, self.d_p, self.N_t, N_pix)
                    psf_basis[p_i, p_j] = np.abs(focal_field).astype(np.float32)**2 #/ norm_contrast
                core_throughput[wl_i, p_i, p_j] = np.sum(circ_mask*psf_basis[p_i, p_j]) / norm_factor
                total_throughput[wl_i, p_i, p_j] = np.sum(psf_basis[p_i, p_j]) / norm_factor
                if i == 0 and j == 0:
                    total_throughput[wl_i, p_i, p_j] = np.sum(on_axis_psf) / norm_factor

            params = np.array([ wl, self.d_p_mas, norm_contrast, norm_factor, N_basis, N_pix, N_pix_overcomplete])
            save_path = data_file_path(self.drm+'_psf_'+pupil_type+'_'+self.d_x_str+'_'+str(int(wl * 1e9))+'.npz','psf')
            np.savez_compressed(save_path, psf_basis=psf_basis / norm_factor, on_axis_psf = on_axis_psf / norm_factor,\
                                no_ss_psf = np.abs(focal_field_no_ss)**2 / norm_factor, params=params)

        save_path_throughput = data_file_path(self.drm+'_throughput_'+pupil_type+'_'+self.d_x_str+'.npz','psf')
        np.savez_compressed(save_path_throughput, core_throughput=core_throughput, total_throughput=total_throughput,\
                            grid_points = psf_points, wl=self.wl_range, d_pix_mas = self.d_p_mas)


    def gen_scene(self, pupil_type, source_field, wl, pupil_symmetry = False):
        """
        Generate the output scene intensity.

        Parameters
        ----------
        pupil_type : str
            The type of pupil used (e.g., circular, hex).
        source_field : np.ndarray
            The input source field array, representing the spatial distribution of the source.
        wl : float
            The wavelength of light in meters.
        pupil_symmetry : bool, optional
            Whether the pupil symmetry should be considered, defaults to False.

        Returns
        -------
        np.ndarray
            A 2D imaged field representing the output intensity.

        Notes
        -----
        - `psf_off_axis` can be much larger than the output `N_p`.
        - `N_s`: Number of source pixels.
        - `N_p`: Number of output pixels.
        """
        psf = np.load(data_file_path(self.drm+'_psf_'+pupil_type+'_'+self.d_x_str+'_'+str(int(wl * 1e9))+'.npz', 'psf'))
        _, _, norm_contrast, norm_factor, N_basis, N_pix, N_pix_overcomplete = psf['params']
        N_basis, N_pix, N_pix_overcomplete = int(N_basis)*(1 + pupil_symmetry) - pupil_symmetry*1, int(N_pix), int(N_pix_overcomplete)
        if pupil_symmetry: psf_basis = self.mirr_symm_psf(psf['psf_basis'], N_basis, N_pix)
        else: psf_basis = psf['psf_basis']

        N_s = np.shape(source_field)[0]
        N_p = int(N_s * self.ratio_s_p)
        N_ghost_p = N_pix + N_p
        output_intensity = np.zeros((N_ghost_p, N_ghost_p), dtype=np.float32)
        suppress_field = trunc_2d(source_field, N_basis)

        psf_points = flat_grid(N_basis)
        psf_basis *= suppress_field[:, :, np.newaxis, np.newaxis]

        for (i, j) in psf_points:
            p_i, p_j = i + N_basis//2, j + N_basis//2
            s_i, s_j = i + N_s//2, j + N_s//2
            if i == 0 and j == 0:
                psf_ij = psf['on_axis_psf'] * source_field[s_i, s_j]
                if N_pix_overcomplete > N_ghost_p:
                    output_intensity += trunc_2d(psf_ij, N_ghost_p)
                else:
                    output_intensity[N_ghost_p//2 - (N_pix_overcomplete//2) : N_ghost_p//2 + (N_pix_overcomplete//2) + 1,\
                    N_ghost_p//2 - (N_pix_overcomplete//2) : N_ghost_p//2 + (N_pix_overcomplete//2) + 1] += psf_ij
            else:
                o_i, o_j = N_ghost_p//2 + i*self.ratio_s_p, N_ghost_p//2 + j*self.ratio_s_p
                output_intensity[o_i - N_pix//2 : o_i + N_pix//2 + 1,  o_j - N_pix//2 : o_j + N_pix//2 + 1] += psf_basis[p_i, p_j]

        output_intensity = trunc_2d(output_intensity, N_p)
        source_field[N_s//2 - N_basis//2 : N_s//2 + N_basis//2 + 1, N_s//2 - N_basis//2 : N_s//2 + N_basis//2 + 1] = 0
        non_iwa_psf = psf['no_ss_psf']
        psf_uniform = bluestein_pad(non_iwa_psf, N_pix_overcomplete, N_pix_overcomplete)
        pad_source = pad_2d(source_field, N_pix_overcomplete*2 - 1)
        output_intensity += trunc_2d(np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(pad_source)*np.fft.fft2(psf_uniform)))), N_p)
        return output_intensity
