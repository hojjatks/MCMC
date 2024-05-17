#%% implementing the MCMC
import numpy as np 
import FNO


#%%
def callikelihood(u,v,sigma,forwardmap):
    # This function is written to calculate the likelihood
    # v is the vector of inputs (In this version, it has three dimensional : [s,x,y])
    # u is distribution of slip on the fault, Assume it is a matrix with size Nx by Nz
    # sigma is the model noise and is scallar
    return np.exp(-0.5*(np.linalg.norm(u-forwardmap(v))**2)/sigma**2)

def MHstep(vk,u,propvar,sigma,p_vk,calculatelikelihood,forwardmap):
    # This function is written to make one MH step
    v_trial = vk + np.sqrt(propvar)*np.random.randn(np.size(vk))
    p_trial=calculatelikelihood(u,v_trial,sigma,forwardmap) # because we assumed the prior is uniform, the posterior is distributed according to likelihood
    a=np.min([1,p_trial/p_vk])
    accept=a>np.random.uniform()
    vkp1=vk
    pvkp1=p_vk
    if accept:
        vkp1=v_trial
        pvkp1=p_trial
    return vkp1,pvkp1,accept

def MCMC(v0,N,u,propvar,sigma,calculatelikelihood,forwardmap):
    # v0 is the initial condition
    # N is the number of MCMC iterations
    samples=np.zeros((N,np.size(v0)))
    vk=v0
    samples[0,:]=v0
    p_vk=calculatelikelihood(u,v0,sigma,forwardmap)

    for index in range(N-1):
        vk=samples[index,:]
        vkp1,pvkp1,accept=MHstep(vk,u,propvar,sigma,p_vk,calculatelikelihood,forwardmap)
        samples[index+1,:]=vkp1
        p_vk=pvkp1


    return samples





# %%
