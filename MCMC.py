# implementing the MCMC
import numpy as np 
import FNO


#%%
def callikelihood(u,v,sigma):
    # This function is written to calculate the likelihood
    # v is the vector of inputs (In this version, it has three dimensional : [s,x,y])
    # u is distribution of slip on the fault, Assume it is a matrix with size Nx by Nz
    # sigma is the model noise and is scallar
    return np.exp(-0.5*(np.norm(u-FNO(v))**2))

def MHstep(vk,var):
    # This function is written to make one MH step
    

    return






