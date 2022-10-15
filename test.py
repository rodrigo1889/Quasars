import matplotlib.pyplot as plt
import numpy as np 
from cosmo import lcdm, wcdm, f1tp

z = np.linspace(0,1)

plt.plot(z,lcdm.H(z,70,0.3),ls="--",color="blue")
plt.plot(z,f1tp.H(z,70,0.3,0),ls="-",color="red")
plt.show()
