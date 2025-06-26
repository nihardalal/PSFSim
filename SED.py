import numpy as np
import os
from glob import glob

class StellarSEDFinder:
    def __init__(self, sed_dir='./data/SEDtemplates/'):
        self.sed_dir = sed_dir
        self.spec_classes = ['o', 'b', 'a', 'f', 'g', 'k', 'm', 'l', 't', 'agb', 'wd']
        self.lum_classes = ['i', 'ii', 'iii', 'iv', 'v', 'vi']  # I=supergiant, V=main sequence
        
        # Pre-load available templates for faster lookup
        self.available_templates = set(glob(os.path.join(sed_dir, 'uk*.dat')) + 
                                      glob(os.path.join(sed_dir, '*_full_fluxed_AA.txt')) +
                                      glob(os.path.join(sed_dir, 'fake_WD_template.txt')))

    def find_sed(self, num, lumclass, interpolate=True):
        """
        Find the SED for a star given its numerical spectral type and luminosity class.
        
        Parameters:
        - num: float, spectral type (e.g., 3.2 = B3.2, 6.5 = K6.5)
        - lumclass: int, luminosity class (1=I, 2=II, ..., 5=V, 6=VI)
        - interpolate: bool, whether to interpolate between subtypes
        
        Returns:
        - (wavelength, flux) tuple
        """
        spec_idx = int(np.floor(num)) - 1
        if spec_idx < 0 or spec_idx >= len(self.spec_classes):
            raise ValueError(f"Spectral type {num} is out of range.")
        
        spec_type = self.spec_classes[spec_idx]
        subtype = int(10 * (num - np.floor(num)))
        lum = self.lum_classes[lumclass - 1] if 1 <= lumclass <= 6 else 'v'  # Default to main sequence
        
        # Handle special cases (L/T dwarfs, white dwarfs)
        if spec_type in ['l', 't', 'wd']:
            return self._load_l_t_wd_sed(spec_type, num)
        
        # Try exact match first
        sed_file = f'uk{spec_type}{subtype}{lum}.dat'
        if os.path.exists(os.path.join(self.sed_dir, sed_file)):
            return self.load_sed(sed_file)
        
        # If not found, find nearest subtype or luminosity class
        if interpolate:
            return self._interpolate_sed(num, lumclass)
        else:
            return self._find_nearest_sed(num, lumclass)

    def _load_l_t_wd_sed(self, spec_type, num):
        """Handle L/T dwarfs and white dwarfs (no luminosity class)."""
        if spec_type == 'wd':
            return self.load_sed('fake_WD_template.txt')
        
        # For L/T dwarfs, match the closest available template
        templates = {
            'l': sorted(glob(os.path.join(self.sed_dir, 'L*_full_fluxed_AA.txt'))),
            't': sorted(glob(os.path.join(self.sed_dir, 'T*_full_fluxed_AA.txt')))
        }
        closest = self._find_closest_template(templates[spec_type], num)
        return self.load_sed(closest)

    def _find_nearest_sed(self, num, lumclass):
        """Fallback to the nearest subtype/luminosity class if exact match fails."""
        spec_type = self.spec_classes[int(np.floor(num)) - 1]
        subtype = int(10 * (num - np.floor(num)))
        lum = self.lum_classes[lumclass - 1]
        
        # Try nearby subtypes (e.g., if K5.3 is missing, try K5 or K6)
        for delta in [0, 1, -1, 2, -2]:  # Check nearby subtypes
            trial_subtype = max(0, min(9, subtype + delta))
            sed_file = f'uk{spec_type}{trial_subtype}{lum}.dat'
            if os.path.exists(os.path.join(self.sed_dir, sed_file)):
                return self.load_sed(sed_file)
        
        # Try other luminosity classes (e.g., fallback to 'v' if 'iii' is missing)
        for trial_lum in [lum, 'v', 'iii', 'i']:  # Common fallbacks
            if trial_lum == lum:
                continue
            sed_file = f'uk{spec_type}{subtype}{trial_lum}.dat'
            if os.path.exists(os.path.join(self.sed_dir, sed_file)):
                print(f"Warning: Using luminosity class {trial_lum} instead of {lum}")
                return self.load_sed(sed_file)
        
        raise FileNotFoundError(f"No suitable template found for type {num} and lum {lumclass}")

    def _interpolate_sed(self, num, lumclass):
        """Interpolate between two nearest templates."""
        spec_type = self.spec_classes[int(np.floor(num)) - 1]
        subtype = 10 * (num - np.floor(num))
        lum = self.lum_classes[lumclass - 1]
        
        # Find bracketing subtypes (e.g., K5.3 -> K5 and K6)
        lower_subtype = int(np.floor(subtype))
        upper_subtype = min(9, lower_subtype + 1)
        weight = subtype - lower_subtype  # 0.3 for K5.3
        
        # Try to load both templates
        lower_file = f'uk{spec_type}{lower_subtype}{lum}.dat'
        upper_file = f'uk{spec_type}{upper_subtype}{lum}.dat'
        
        try:
            wav1, flux1 = self.load_sed(lower_file)
            wav2, flux2 = self.load_sed(upper_file)
        except FileNotFoundError:
            return self._find_nearest_sed(num, lumclass)  # Fallback if interpolation fails
        
        if not np.array_equal(wav1, wav2):
            raise ValueError("Wavelength grids differ; cannot interpolate.")
        
        # Linear interpolation in log-space (better for stellar spectra)
        log_flux1 = np.log10(flux1)
        log_flux2 = np.log10(flux2)
        interp_log_flux = (1 - weight) * log_flux1 + weight * log_flux2
        interp_flux = 10**interp_log_flux
        
        return (wav1, interp_flux)

    def load_sed(self, sed_file):
        """Load a single SED file."""
        path = os.path.join(self.sed_dir, sed_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"SED file {sed_file} not found.")
        
        # Handle different file formats (UKIRT vs. L/T dwarf)
        if sed_file.startswith('uk'):
            wav, flux = np.loadtxt(path, skiprows=3, unpack=True)
        else:  # L/T dwarf or WD
            wav, flux = np.loadtxt(path, unpack=True)
        
        return (wav, flux)

    def _find_closest_template(self, templates, num):
        """Find the closest L/T dwarf template (e.g., L5.2 -> L5)."""
        subtypes = []
        for t in templates:
            # Extract subtype from filename (e.g., 'L5_2MASS...' -> 5.0)
            parts = os.path.basename(t).split('_')[0].lower()
            if parts.startswith(('l', 't')):
                subtype = float(parts[1:].replace('p', '.').replace('m', '-'))
                subtypes.append(subtype)
        
        closest_idx = np.argmin(np.abs(np.array(subtypes) - num))
        return templates[closest_idx]