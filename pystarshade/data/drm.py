import numpy as np

# Put telescope design reference missions in here
# Note: do not have to stick to these parameters

mas_to_rad = 4.84814e-9
wfirst_wl_bands = [[425, 552], [606, 787], [747, 970]]

telescope_params = {
    "wfirst": {
        "focal_length_lens": 30,
        "diameter_telescope_m": 2.36,
        "radius_lens": 2.36 / 2,
        "dist_ss_t": 3.724225668350351e07,
        "iwa_mas": 72,
        "num_pet": 16,
        "ss_radius": 13,
        "grey_mask_dx": ["01m", "005m", "002m", "001m", "0005m"],
        "dx_": [0.01, 0.005, 0.002, 0.001, 0.0005],
        "wl_bands": wfirst_wl_bands,
        "dist_ss_t": 3.724225668350351e07
        * np.sum(wfirst_wl_bands[0])
        / np.sum(wfirst_wl_bands, axis=1),
        "iwa": 72
        * np.sum(wfirst_wl_bands, axis=1)
        / np.sum(wfirst_wl_bands[0]),
    },
    "HWO": {
        "focal_length_lens": 10,
        "diameter_telescope_m": 6,
        "radius_lens": 3,
        "ss_radius": 30,
        "dist_ss_t": [9.52e07],
        "num_pet": 24,
        "wl_bands": [[500, 1000]],
        "dx_": [0.04, 0.001],
        "grey_mask_dx": ["04m", "001m"],
        "iwa": [30 / 9.52e07 / mas_to_rad],  # R_ss / dist_ss_t in MAS
    },
    "habex": {
        "focal_length_lens": 11,
        "diameter_telescope_m": 4,
        "radius_lens": 3,
        "ss_radius": 36,
        "dist_ss_t": [1.24e08],
        "num_pet": 24,
        "wl_bands": [[300, 1000]],
        "dx_": [0.01, 0.005, 0.002, 0.001, 0.0005],
        "grey_mask_dx": ["01m", "005m", "002m", "001m", "0005m"],
        "iwa": [60],  # R_ss / dist_ss_t in MAS
    },
    "hwo": {
        "focal_length_lens": 10,
        "diameter_telescope_m": 6,
        "radius_lens": 3,
        "ss_radius": 30,
        "dist_ss_t": [9.52e07],
        "num_pet": 24,
        "wl_bands": [[500, 600]],
        "dx_": [0.04, 0.001],
        "grey_mask_dx": ["04m", "001m"],
        "iwa": [30 / 9.52e07 / mas_to_rad],  # R_ss / dist_ss_t in MAS
    },
}
