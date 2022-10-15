import numpy as np 
from multiprocess import Pool
import emcee 
from call_data import CC, PANTHEON_BINNED, BAO_DM, BAO_roverDV, BAO_DV, QUASARS_BINNED2, GRB
from cosmo import lcdm
import main 

#Calling data only one time to make the code more efficient
z_obs, H_obs,eH_obs = CC()
z_p, mb_p,emb_p = PANTHEON_BINNED()
z_b1, dm_1,edm_1 = BAO_DM()
z_b2, bao2,ebao2 = BAO_roverDV()
z_b3, bao3,ebao3,r_fid3 = BAO_DV()
z_qso, mu_qso,emu_qso = QUASARS_BINNED2()
z_grb, mu_grb, emu_grb = GRB()

data = ([z_obs,z_p,z_b1,z_b2,z_b3,z_qso,z_grb], [H_obs,mb_p,dm_1,bao2,bao3,mu_qso,mu_grb],[eH_obs,emb_p,edm_1,ebao2,ebao3,emu_qso,emu_grb])
#data = ([z_obs,z_p,z_qso], [H_obs,mb_p,mu_qso],[eH_obs,emb_p,emu_qso])
def lnlike_cc(theta, x, y, yerr):
    H0,m,M = theta
    #H0,m = theta
    return -0.5*np.sum(((y[0] - lcdm.H(x[0],H0,m))/yerr[0])**2+np.log(yerr[0]**2))
def lnlike_pantheon(theta,x,y,yerr):
    H0,m,M = theta
    teory = lcdm.distance_modulus(x[1],H0,m) - M 
    return -0.5*np.sum(((y[1] - teory)/yerr[1])**2+np.log(yerr[1]**2))
def lnlike_BAOdm(theta,x,y,yerr):
    H0,m,M = theta
    theory = lcdm.DMoverrd(x[2],H0,m,r_fid=147.78)
    return -0.5*np.sum(((y[2] - theory)/yerr[2])**2)
def lnlike_BAOroverDV(theta,x,y,yerr):
    H0,m,M = theta
    theory = lcdm.rdoverDV(x[3],H0,m)
    return -0.5*((y[3]-theory)/(yerr[3])**2)
def lnlike_BAO_DV(theta,x,y,yerr):
    H0,m,M = theta
    theory0 = lcdm.DVoverrd(x[4][0],H0,m,r_fid=r_fid3[0])
    theory1 = lcdm.DVoverrd(x[4][1],H0,m,r_fid=r_fid3[1])
    lkk0 = -0.5*(((y[4][0] - theory0)/yerr[4][0])**2)
    lkk1 = -0.5*(((y[4][1] - theory1)/yerr[4][1])**2)
    return lkk0 + lkk1
def lnlike_BAO(theta,x,y,yerr):
    return lnlike_BAOdm(theta,x,y,yerr) + lnlike_BAO_DV(theta,x,y,yerr) + lnlike_BAOroverDV(theta,x,y,yerr)
def lnlike_Quasars(theta,x,y,yerr):
    H0,m,M = theta
    theory = lcdm.distance_modulus(x[5],H0,m)
    return -0.5*np.sum(((y[5] - theory)/yerr[5])**2+np.log(yerr[5]**2))
def lnlike_GRB(theta,x,y,yerr):
    H0,m,M = theta
    theory = lcdm.distance_modulus(x[6],H0,m)
    return -0.5*np.sum(((y[6] - theory)/yerr[6])**2)

def lnlike(theta,x,y,yerr):
    return lnlike_cc(theta,x,y,yerr) + lnlike_pantheon(theta,x,y,yerr) + lnlike_Quasars(theta,x,y,yerr)


def lnprior(theta):
    H0,om,M = theta
    #H0,m = theta
    if 50 < H0 < 100 and 0.0 < om < 0.9 and 10 < M < 30:
    #if 50.0 < H0 < 100.0 and 0.0 < m < 0.5: 
        return 0.0
    return -np.inf
def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)
initial = np.array([65,0.35,19])
nwalkers=35
p0 = [np.array(initial) + 1e-5 * np.random.randn(len(initial)) for i in range(nwalkers)]

N = 10000
sampler, pos, prob, state = main.main(p0,nwalkers,N,len(initial),lnprob,data)

samples = sampler.flatchain
np.savetxt("Chains/lcdm/cc_pantheon_bao_qsob.txt",samples,newline="\n")
