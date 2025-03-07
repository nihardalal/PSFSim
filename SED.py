import numpy as np
import os

def find_sed(num, lumclass):
    specList = ['o', 'b', 'a', 'f', 'g', 'k', 'm', 'agb', 'wd']
    specType = specList[int(np.floor(num)-1)]
    subType = int(np.floor(10*(num-np.floor(num))))
    interp = (abs(np.floor(10*(num-np.floor(num))) - 10*(num-np.floor(num)))>0.001)
    lumlist = ['i', 'ii', 'iii', 'iv', 'v', 'vi']
    lum = lumlist[lumclass-1]
    sedFileName = 'uk'+specType+str(subType)+lum+'.dat'
    sedFileName2 = 'uk'+specType+str(subType+1)+lum+'.dat'
    if not interp:
        return load_sed(sedFileName)
    else:
        return interp_sed(sedFileName, sedFileName2)
    
def load_sed(sedFileName):
    path = './data/SEDtemplates'
    path_to_file_name = path+sedFileName
    if os.path.exists(path_to_file_name):
        wav = np.loadtxt(path_to_file_name, skiprows=3, usecols = 0)
        spec = np.loadtxt(path_to_file_name, skiprows=3, usecols = 1)
        return (wav, spec)
    else:
        raise Exception("File doesn't exist, need to write routine to use nearby SED template")

def interp_sed(sedFileName1, sedFileName2):
    wav1, spec1 = load_sed(sedFileName1)
    wav2, spec2 = load_sed(sedFileName2)

    if wav1==wav2:
        #Need more sophisticated interpolation based on Teff/metalicity later. Ideas are to do a weighted mean
        #based on temperature of current star or something so it gets more accurate than just a mean of fluxes - ND
        return (wav1, np.mean(spec1, spec2))
    else: 
        raise Exception("Need more complicated interpolation routine")




