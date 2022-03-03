# MT matrix format:
    # 1 row per bead per frame, sorted by frame number then x position (roughly)
    # columns:
    # 1:2 - X and Y positions (in pixels)
    # 3   - Integrated intensity(mass)
    # 4   - Rg squared of feature
    # 5   - eccentricity
    # 6   - frame #
    # 7   - time of frame
#dataframe format used in trackpy    
# df = pd.DataFrame(data, columns = ['y', 'x', 'mass', 'size', 'ecc', signal , raw_mass, 'ep', 'frame'])     

        
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import plot, draw, show
from pandas import DataFrame, Series
import pims
import trackpy as tp
import time
import threading
import multiprocessing
import concurrent.futures

ps  = 10
dispDistance = (ps+1)/2                 # Distance a particle is allowed to move between 2 consecutive frames 

if __name__ == "__main__":
    fname  = '/home/vijayakumar/Sagar/python/correlation/featureFinding/fov0/MT_featsize_11.npy'
    data = np.load(fname)
    
    # Taking goodEnough such that we consider particles in all the frames.
    # Last frame - staring frame + 1
    # Be careful about which column of the data contains frame information 
    goodEnough = data[:,5][len(data[:,5])-1] - data[:,5][0] + 1

    print("Starting linking particles with ps = ",ps," goodenough = ", goodEnough)
    
    df = pd.DataFrame({'y':data[:,1], 'x':data[:,0], 'mass':data[:,2],'frame':data[:,5], 'time': data[:,6]})

    t = tp.link(df, dispDistance)
    t2 = tp.filter_stubs(t, goodEnough)
    traj = tp.compute_drift(t2)
    t1 = tp.subtract_drift(t2,traj)

    column0 = t1['x'].tolist()
    column1 = t1['y'].tolist()  
    column2 = t1['frame'].tolist()
    column3 = t1['particle'].tolist()
    
    finalData = [column0, column1, column2, column3]    
    finalData = np.array(finalData)

    finalData = np.transpose(finalData)
    np.save('lp57', finalData)