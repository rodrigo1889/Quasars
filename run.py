import numpy as np 
from multiprocess import Pool
import emcee 
from call_data import CC, PANTHEON_BINNED, BAO_DM, BAO_roverDV, BAO_DV, QUASARS, GRB
from cosmo import f1tp
import main 

#Calling data only one time to make the code more efficient
z_obs, H_obs,eH_obs = CC()
z_p, mb_p,emb_p = PANTHEON_BINNED()
z_b1, dm_1,edm_1 = BAO_DM()
z_b2, bao2,ebao2 = BAO_roverDV()
z_b3, bao3,ebao3,r_fid3 = BAO_DV()
z_qso, mu_qso,emu_qso = QUASARS()
z_grb, mu_grb, emu_grb = GRB()

data = ([z_obs,z_p,z_b1,z_b2,z_b3,z_qso,z_grb], [H_obs,mb_p,dm_1,bao2,bao3,mu_qso,mu_grb],[eH_obs,emb_p,edm_1,ebao2,ebao3,emu_qso,emu_grb])
#data = ([z_obs,z_p,z_qso], [H_obs,mb_p,mu_qso],[eH_obs,emb_p,emu_qso])
def lnlike_cc(theta, x, y, yerr):
    H0,m,M,b = theta
    #H0,m,b = theta
    return -0.5*np.sum(((y[0] - f1tp.H(x[0],H0,m,b))**2/yerr[0]**2 + np.log(yerr[0]**2)))
def lnlike_pantheon(theta,x,y,yerr):
    #H0,m,M,b = theta
    H0,m,b = theta
    teory = f1tp.distance_modulus(x[1],H0,m,b) - M 
    return -0.5*np.sum(((y[1] - teory)**2/yerr[1]**2 +np.log(yerr[1]**2)))
def lnlike_BAOdm(theta,x,y,yerr):
    #H0,m,M,b = theta
    H0,m,b = theta
    theory = f1tp.DMoverrd(x[2],H0,m,b,r_fid=147.78)
    return -0.5*np.sum(((y[2] - theory)/yerr[2])**2)
def lnlike_BAOroverDV(theta,x,y,yerr):
    #H0,m,M,b = theta
    H0,m,b = theta
    theory = f1tp.rdoverDV(x[3],H0,m,b)
    return -0.5*((y[3]-theory)/(yerr[3])**2)
def lnlike_BAO_DV(theta,x,y,yerr):
    #H0,m,M,b = theta
    H0,m,b = theta
    theory0 = f1tp.DVoverrd(x[4][0],H0,m,b,r_fid=r_fid3[0])
    theory1 = f1tp.DVoverrd(x[4][1],H0,m,b,r_fid=r_fid3[1])
    lkk0 = -0.5*(((y[4][0] - theory0)/yerr[4][0])**2)
    lkk1 = -0.5*(((y[4][1] - theory1)/yerr[4][1])**2)
    return lkk0 + lkk1
def lnlike_BAO(theta,x,y,yerr):
    return lnlike_BAOdm(theta,x,y,yerr) + lnlike_BAO_DV(theta,x,y,yerr) + lnlike_BAOroverDV(theta,x,y,yerr)
def lnlike_Quasars(theta,x,y,yerr):
    H0,m,M,b = theta
    theory = f1tp.distance_modulus(x[2],H0,m,b)
    return -0.5*np.sum(((y[2] - theory)/yerr[2])**2)
def lnlike_GRB(theta,x,y,yerr):
    H0,m,M,b = theta
    theory = f1tp.distance_modulus(x[6],H0,m,b)
    return -0.5*np.sum(((y[6] - theory)/yerr[6])**2)

def lnlike(theta,x,y,yerr):
    return lnlike_BAO(theta,x,y,yerr)


def lnprior(theta):
    #H0,m,M,b = theta
    H0,m,b = theta
    #if 50 < H0 < 100 and 0.1 < m < 0.5 and 10 < M < 30 and -1 <= b <= 1:
    if 50.0 < H0 < 80.0 and 0.0 < m < 1.0 and -1.0 <= b <= 1.0:
        return 0.0
    return -np.inf
def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)
initial = np.array([65,0.35,0.1])
#initial = np.array([65,0.35,0.1])
nwalkers=15
p0 = [np.array(initial) + 1e-5 * np.random.randn(len(initial)) for i in range(nwalkers)]

sampler, pos, prob, state = main.main(p0,nwalkers,10000,len(initial),lnprob,data)

samples = sampler.flatchain
np.savetxt("Chains/Power-Law/bao.txt",samples,newline="\n")
