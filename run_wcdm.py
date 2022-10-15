import numpy as np 
from multiprocess import Pool
import emcee 
from call_data import PANTHEON_BINNED, PANTHEONPLUS
from cosmo import wcdm
import main 


z_p, mb_p,emb_p = PANTHEON_BINNED()
zcmb, zhel, mb, emb, mu, emu = PANTHEONPLUS()

data = ([z_p,zcmb],[mb_p,mb],[emb_p,emb])
def lnlike_pantheon(theta,x,y,yerr):
    m,w,M = theta
    teory = wcdm.distance_modulus(x[0],74.0,m,w) - M 
    return -0.5*np.sum(((y[0] - teory)**2/yerr[0]**2 +np.log(yerr[0]**2)))
    
def lnlike_pantheonplus(theta,x,y,yerr):
    m,w,M = theta
    teory = wcdm.distance_modulus(x[1],74.0,m,w) - M 
    return -0.5*np.sum(((y[1] - teory)**2/yerr[1]**2 +np.log(yerr[1]**2)))
    
    
def lnlike(theta,x,y,yerr):
    return lnlike_pantheon(theta,x,y,yerr)

def lnprior(theta):
    m,w,M = theta
    if 0.0 < m < 0.9 and -2.0 < w <= 1.0 and 10 < M < 30:
    #if 50.0 < H0 < 100.0 and 0.0 < m < 0.5: 
        return 0.0
    return -np.inf
def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)
initial = np.array([0.35,-1.0,19])
nwalkers=35
p0 = [np.array(initial) + 1e-5 * np.random.randn(len(initial)) for i in range(nwalkers)]

sampler, pos, prob, state = main.main(p0,nwalkers,5000,len(initial),lnprob,data)

samples = sampler.flatchain
np.savetxt("Chains/Pantheon-wcdm/pantheon.txt",samples,newline="\n")
