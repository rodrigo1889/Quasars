import numpy as np 
from scipy import integrate

def H(z,H0,m): #Same units as H0 (km/s/Mpc)
    return H0*np.sqrt(m*(1+z)**3+(1-m))
def E(z,m): #Without units
    return 1/np.sqrt(m*(1+z)**3+(1-m))
def D_c(z,m): #Without units
    return integrate.quad(E,0,z,args=(m,))[0]
D_c = np.vectorize(D_c)
def D_M(z,H0,m):# in Mpc
    return (299792.46/H0)*D_c(z,m)
def D_L(z,H0,m):#In Mpc
    return (1+z)*D_M(z,H0,m)
def distance_modulus(z,H0,m): #No units thus the therm +25 is associated to Mpc
    return 5*np.log10(D_L(z,H0,m))+25
def D_V(z,H0,m): #In Mpc
    c = 299792.46
    return (((c*z)/(H(z,H0,m)))*(D_M(z,H0,m)**2))**(1/3)    
def c_s(z): # in km/s
    ogam = 2.469e-5
    obar = 0.0224
    c = 299792.46
    R = ((3*obar)/(4*ogam))/(1+z)
    return c/np.sqrt(3*(1+R))
def zd(H0,m): #z drag 
    h = H0/100
    wm = m*h**2
    obh = 0.0224
    b1 = (0.313*wm**(-0.419))*(1+0.607*wm**0.674)
    b2 = 0.238*wm**(0.223)
    arriba = 1291*wm**(0.251)
    abajo = 1+0.659*wm**0.828
    mult = (1 + b1*obh**b2)
    zd = (arriba/abajo)*mult
    return zd
def r_d(H0,m): #in Mpc 
    """
    Using the aproximation in Mpc given by Eistenstein & Hu (1998)
    """
    wb = 0.0224
    Tcmb = 2.75 #in k
    h = H0/100
    wm = m*0.7**2
    zeq = 2.50e4*wm*(Tcmb/2.7)**(-4)
    keq = 7.46e-2*wm*(Tcmb/2.7)**(-2)
    Rd = 31.5*wb*(Tcmb/2.7)**(-4)*(1e3/zd(H0,m))
    Req = 31.5*wb*(Tcmb/2.7)**(-4)*(1e3/zeq)
    return 2./3./keq*(6./Req)**0.5 *np.log((np.sqrt(1 + Rd) + np.sqrt(Rd+Req))/(1 + Req**0.5))
def DMoverrd(z,H0,m,r_fid = 147.78):
    return D_M(z,H0,m)*(r_fid/r_d(H0,m))
def DVoverrd(z,H0,m,r_fid=147.78):
    return D_V(z,H0,m)*(r_fid/r_d(H0,m))
def Hoverrd(z,H0,m,r_fid=147.78):
    return H(z,H0,m)*(r_fid/r_d(H0,m))
def rdoverDV(z,H0,m):
    return r_d(H0,m)/D_V(z,H0,m)
#def r_d2(H0,m):
#    h = H0/100
#    wm = m*h**2;wb = 0.0224
#    a1 = 0.00785436; a2 = 0.177084; a3 = 0.00912388
#    a4 = 0.618711; a5 = 11.9611; a6 = 2.81343; a7 = 0.784719
#    return 1/(a1*wb**a2+a3*wm**a4+a5*wb**a6*wm**a7)
#def r_d(z,H0,m):
#    integral = lambda z: c_s(z)/H(z,H0,m)
#    return integrate.quad(integral,zd(H0,m),np.inf)[0]

