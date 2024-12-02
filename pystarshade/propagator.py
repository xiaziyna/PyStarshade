from data.drm import telescope_params, mas_to_rad
import glob
import numpy as np
from simulate_field import source_field_to_pupil, pupil_to_ccd
from pupil import make_pupil
from util import flat_grid, bluestein_pad, trunc_2d, pad_2d

class StarshadeProp:
    """
    source angular resolution, default = 2mas
    drm : 'wfirst' or 'hwo'
    
    Use: 
        Pick DRM
        This class will generate the necessary files, 
        suppress_region is the angular region where starlight is suppressed
    Example:
        hwo_starshade = StarshadeProp(drm = 'hwo')
        hwo_starshade.gen_pupil_field()
        hwo_starshade.gen_psf_basis(pupil_type = 'hex')

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
        '''
        Load the design reference mission (defined inside data/telescope_drm.py)
        '''
        drm_params = telescope_params[drm]
        self.f = drm_params['focal_length_lens']
        self.suppress_region = int(2 * drm_params['iwa'][band_i])
        self.suppress_region += self.suppress_region%2
        self.r_lens = drm_params['radius_lens']
        self.mask_choice = mask_choice

        self.N_x =  int(2*(drm_params['ss_radius']+.2)*10 / (drm_params['dx_'][mask_choice]*10)) + 1
        self.d_x = drm_params['dx_'][mask_choice]
        self.d_x_str = drm_params['grey_mask_dx'][mask_choice]
        self.ss_mask_fname_partial = '../../mask/grey_%s_%s_mask_%s' % (drm, str(drm_params['num_pet']), self.d_x_str) #change my path please 

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

    def calc_magnification(self, dist_xo_ss):
        return self.f / (dist_xo_ss + self.dist_ss_t)

    def calc_d_s(self, d_s_mas, dist_xo_ss):
        return (d_s_mas * (dist_xo_ss * mas_to_rad))

    def calc_d_s_mas(self, d_s, dist_xo_ss):
        return (d_s / dist_xo_ss) / mas_to_rad

    def gen_pupil_field(self, chunk = 1):
        """
        Generates the field at the pupil for choice of starshade
        """
        if glob.glob(f"data/fields/{self.drm}_pupil_{self.d_x_str}*.npz"): return

        print("File does not exist. Generating.")
        if chunk:
            ss_mask_fname = self.ss_mask_fname_partial + '.dat'
        else:
            ss_mask_fname = self.ss_mask_fname_partial + 'qu.npz'
        over_N_t = 2*int((self.suppress_region // self.d_t_mas) + 100)
        over_N_t += (1-over_N_t%2)
        for wl_i in self.wl_range:
            field_incident_telescope, field_free_prop, params = source_field_to_pupil(ss_mask_fname, wl_i,\
            self.dist_ss_t, N_x = self.N_x, N_t = over_N_t, dx = self.d_x, dt = self.d_t, chunk=chunk)
            np.savez_compressed('data/fields/'+self.drm+'_pupil_'+self.d_x_str+'_'+str(int(wl_i * 1e9))+'.npz',\
            field=field_incident_telescope, freesp_field=field_free_prop, params=params)

    def gen_pupil(self, pupil_type):
        """
        Options for pupil_type (generated with HCIPy): 
        circ, ELT, GMT, TMT, Hale, LUVOIR-A, LUVOIR-B, Magellan, VLT, HiCAT, HabEx, HST, JWST, Keck, hex
        """
        try:
            pupil_data = np.load('data/pupils/'+pupil_type+'_'+str(int(self.N_t))+'.npz')
            self.pupil_mask = pupil_data['pupil']
        except FileNotFoundError:
            print(f"File not found. Generating a new pupil mask.")
            make_pupil(self.N_t, pupil_type)
            pupil_data = np.load('data/pupils/'+pupil_type+'_'+str(int(self.N_t))+'.npz')
            self.pupil_mask = pupil_data['pupil']

    def gen_psf_basis(self, pupil_type, pupil_symmetry = False):
        """
        Generates the incoherent field at the pupil for choice of starshade
        Contrast is measured in units of if no starhade were present, i.e. just FFT of the pupil_mask 
        If the pupil is symmetric, only generate the positive quadrant of PSF's
        """
        if glob.glob(f"data/psf/{self.drm}_psf_{pupil_type}_{self.d_x_str}*.npz"): return

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

        grid_points = flat_grid(N_basis, negative = 1 - pupil_symmetry)

        for wl_i in range(self.N_wl):
            psf_basis = np.zeros((N_basis, N_basis, N_pix, N_pix), dtype=np.float32)
            on_axis_psf = np.zeros((N_pix_overcomplete, N_pix_overcomplete), dtype=np.float64)
            data = np.load('data/fields/'+self.drm+'_pupil'+'_'+self.d_x_str+'_'+str(int(self.wl_range[wl_i]*1e9))+'.npz')
            wl = data['params'][0]
            pupil_field = data['field']
            over_N_t = int(np.shape(pupil_field)[0])
            pupil_field_no_ss = trunc_2d(data['freesp_field'], self.N_t)
            focal_field_no_ss = pupil_to_ccd(wl, self.f, pupil_field_no_ss, self.pupil_mask, self.d_t, self.d_p, self.N_t, N_pix_overcomplete)
            norm_contrast = np.max(np.abs(focal_field_no_ss)**2)
            norm_factor = np.sum(np.abs(focal_field_no_ss)**2)
            circ_mask = np.hypot(x, y) <= self.ang_res_pix[wl_i]

            for (i, j) in grid_points:
                p_i, p_j = i + N_basis//2, j + N_basis//2
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

            params = np.array([ wl, self.d_p_mas, norm_contrast, N_basis, N_pix, N_pix_overcomplete])
            np.savez_compressed('data/psf/'+self.drm+'_psf_'+pupil_type+'_'+self.d_x_str+'_'+str(int(wl * 1e9))+'.npz',\
                                psf_basis=psf_basis / norm_contrast, on_axis_psf = on_axis_psf / norm_contrast,\
                                no_ss_psf = np.abs(focal_field_no_ss)**2 / norm_contrast, params=params)

        np.savez_compressed('data/psf/'+self.drm+'_throughput_'+pupil_type+'_'+self.d_x_str+'.npz',\
        core_throughput=core_throughput, total_throughput=total_throughput, grid_points = grid_points,\
        wl=self.wl_range, d_pix_mas = self.d_p_mas)

    def gen_scene(self, pupil_type, source_field, wl, pupil_symmetry = False):
        '''
        Generate the scene intensity based on the specified pupil type, source field, wavelength, 
        and pupil symmetry.
        Args:
            pupil_type : str
                The type of pupil used (e.g., circular, hex).
            source_field : np.ndarray
                The input source field array, representing the spatial distribution of the source.
            wl : float
                The wavelength of light in meters.
            pupil_symmetry : bool, optional
                Whether the pupil symmetry should be considered. Default is False.

        Returns:
            output_intensity: 2D imaged field

        psf_off_axis can be much larger than output N_p
        N_p : number of output pixels

        '''
        psf = np.load('data/psf/'+self.drm+'_psf_'+pupil_type+'_'+self.d_x_str+'_'+str(int(wl * 1e9))+'.npz')
        _, _, norm_contrast, N_basis, N_pix, N_pix_overcomplete = psf['params']
        psf_basis = psf['psf_basis']
        N_basis, N_pix, N_pix_overcomplete = int(N_basis), int(N_pix), int(N_pix_overcomplete)
        N_s = np.shape(source_field)[0]
        N_p = int(N_s * self.ratio_s_p)
        output_intensity = np.zeros((N_p, N_p), dtype=np.float32)
        suppress_field = trunc_2d(source_field, N_basis)
        grid_points = flat_grid(N_basis, negative = 1 - pupil_symmetry)
        psf_basis *= suppress_field[:, :, np.newaxis, np.newaxis]

        for (i, j) in grid_points:
            p_i, p_j = i + N_basis//2, j + N_basis//2
            s_i, s_j = i + N_s//2, j + N_s//2
            if i == 0 and j == 0:
                psf_ij = psf['on_axis_psf'] * source_field[s_i, s_j]
                if N_pix_overcomplete > N_p:
                    output_intensity += trunc_2d(psf_ij, N_p)
                else:
                    output_intensity[N_p//2 - (N_pix_overcomplete//2) : N_p//2 + (N_pix_overcomplete//2) + 1,\
                    N_p//2 - (N_pix_overcomplete//2) : N_p//2 + (N_pix_overcomplete//2) + 1] += psf_ij
            else:
                o_i, o_j = N_p//2 + i*self.ratio_s_p, N_p//2 + j*self.ratio_s_p
                output_intensity[o_i - N_pix//2 : o_i + N_pix//2 + 1,  o_j - N_pix//2 : o_j + N_pix//2 + 1] += psf_basis[p_i, p_j]

        source_field[N_s//2 - N_basis//2 : N_s//2 + N_basis//2 + 1, N_s//2 - N_basis//2 : N_s//2 + N_basis//2 + 1] = 0
        non_iwa_psf = psf['no_ss_psf']
        psf_uniform = bluestein_pad(non_iwa_psf, N_pix_overcomplete, N_pix_overcomplete)
        pad_source = pad_2d(source_field, N_pix_overcomplete*2 - 1)
        output_intensity += trunc_2d(np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(pad_source)*np.fft.fft2(psf_uniform)))), N_p)
        return output_intensity

#hwo_starshade = StarshadeProp(drm = 'hwo')
#hwo_starshade.gen_pupil_field()
#hwo_starshade.gen_psf_basis('hex')
#test_field = np.zeros((1001, 1001))
#test_field[500, 500] = 1e11
#test_field[800, 760] = 1
#out = hwo_starshade.gen_scene('hex', test_field, 500e-9)
#np.savez_compressed('test_field.npz', field = out)
