import emcee 
import numpy as np 
from multiprocess import Pool

def main(p0,nwalkers,niter,ndim,lnprob,data):
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data,pool=pool)

        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 100,progress=True)
        sampler.reset()

        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter,progress=True)

    return sampler, pos, prob, state