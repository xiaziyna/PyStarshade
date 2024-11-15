class Pupil:
    def __init__(self, filename=None, pupil_type=None, R_p=None, d_s_mas = None, ):
        """
        Initializes the Pupil class.
        Either pass a predefined pupil or generate a new one.
        Pupil sampling should be an integer fraction of the source resolution in mas
        unless you're using coherent diffraction. 
        
        Parameters:
        - create_new: bool, if True, creates a new pupil; if False, loads from file.
        - filename: str, the file path to load the pupil from (used if create_new is False).
        - size: tuple, the dimensions of the pupil to create if creating a new one.
        """

        self.filename = filename
        self.pupil_data = None 
        if self.filename is None: 
            self.create_pupil(pupil_type)
        self.pupil_symmetry = 0
        if pupil_type == 'circ' or 'hex': self.pupil_symmetry = 1

        else:
            self.d_p_mas = 
            factor_source_pupil = d_s_mas/d_p_mas 
            if np.abs(factor_source_pupil%1):
                from scipy.ndimage import zoom:
                scale_factor = d_x / d_x_new

            self.load_pupil()

    def create_pupil(self):
        """
        Creates a new pupil with specified dimensions.
        """
        # Placeholder code for creating a new pupil
        # Example: Initialize self.pupil_data as a 2D array representing the pupil
        pass

    def interp_pupil(self):
        """
        Loads an existing pupil from a file.
        """
        # Placeholder code for loading pupil data from a file
        pass



# need to create template set of PSF's with 

# First things first, lets reconcile the fresnel methods!

