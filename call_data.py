import pandas as pd 
import numpy as np 

### FULL DATA SETS

def CC():
    data = pd.DataFrame(pd.read_csv("Full_data/cc.csv"))
    z = data["z"].to_numpy(); H = data["H(z)"].to_numpy(); Herr = data["s_H"].to_numpy()
    return z,H,Herr

def PANTHEON():
    data = pd.DataFrame(pd.read_csv("Full_data/Pantheon.txt",header=None,sep=" "))
    data = data.sort_values(by=1)
    zcmb = data[1].to_numpy(); mb = data[4].to_numpy(); emb = data[5].to_numpy()
    return zcmb,mb,emb

def QUASARS():
    data = pd.DataFrame(pd.read_csv("Full_data/Quasars.csv"))
    z = data["z"].to_numpy(); mu = data["DM"].to_numpy(); emu = data["e_DM"].to_numpy()
    return z,mu,emu

def BAO_DM(): #Arxiv 2111.02420
    data = pd.DataFrame(pd.read_csv("Full_data/Baodm.txt",header=None))
    z = data[0].to_numpy(); DM = data[1].to_numpy(); eDM = data[2].to_numpy()
    return z,DM,eDM

def BAO_roverDV():
    return  0.106,0.336, 0.015

def BAO_DV():
    return np.array([0.15,1.52]),np.array([664,3843]),np.array([25,147]),np.array([148.69,147.78])

def BAO_H():
    data = pd.DataFrame(pd.read_csv("Full_data/BaoH.txt",header=None))
    z = data[0].to_numpy(); H = data[1].to_numpy(); eH = data[2].to_numpy()
    return z,H,eH

def GRB():
    data = pd.DataFrame(pd.read_csv("Full_data/grb.csv"))
    z = data["Redshift"].to_numpy(); mu = data["μ"].to_numpy(); emu = data["σ μ"].to_numpy()
    return z,mu,emu

def PANTHEONPLUS():
    data = pd.DataFrame(pd.read_csv("Full_data/Pantheonplus.txt"))
    data = data.sort_values(by="zCMB")
    zcmb = data["zCMB"].to_numpy(); zhel = data["zHEL"].to_numpy(); mb = data["mB"].to_numpy(); emb = data["mBERR"].to_numpy(); mu = data["MU_SH0ES"].to_numpy(); emu = data["MU_SH0ES_ERR_DIAG"].to_numpy()
    return zcmb, zhel, mb, emb, mu, emu



### BINNED DATA SETS 

def PANTHEON_BINNED():
    data = pd.DataFrame(pd.read_csv("Binned_data/Pantheon.txt",header=None,sep=" "))
    data = data.sort_values(by=1)
    zcmb = data[1].to_numpy(); mb = data[4].to_numpy(); emb = data[5].to_numpy()
    return zcmb,mb,emb

def QUASARS_BINNED():
    data = pd.DataFrame(pd.read_csv("Binned_data/Quasars.txt",header=None))
    data = data.transpose()
    z = data[0].to_numpy(); mu = data[1].to_numpy(); emu = data[2].to_numpy()
    return z,mu,emu


def QUASARS_BINNED2():
    data = np.genfromtxt("Binned_data/Quasars_2.txt")
    z = data[0]; dm = data[1]; edm = data[2]
    return z, dm, edm 



