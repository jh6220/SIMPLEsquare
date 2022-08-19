import pandas as pd
from scipy.interpolate import LinearNDInterpolator
import numpy as np
df = pd.read_csv(r"C:\Users\jakub\OneDrive\Documents\JupyterNotebook\UROP\DataCFD\ldc2d-re100-all\all\5000NUcav.plt",delim_whitespace=True)
data = df.to_numpy()
points = df.iloc[5000:5000+2601,0:2].to_numpy()
df = pd.read_csv(r"C:\Users\jakub\OneDrive\Documents\JupyterNotebook\UROP\DataCFD\ldc2d-re100-all\all\5000NUcav.var",delim_whitespace=True,header=None)
vals = df.to_numpy()
vals
interp = LinearNDInterpolator(points, vals)

# x-middle slice u-velocity
ix_mid = int(np.floor(u.shape[0]/2))
x_slice = prob.X_u[ix_mid,:]
y_slice = prob.Y_u[ix_mid,:]
u_slice = u[ix_mid,:]
u_slice_interp = interp(x_slice,y_slice)[:,0]
plt.plot(y_slice,u_slice)
plt.plot(y_slice,u_slice_interp,'x')
plt.legend(["Ghia et al 1982","N="+str(prob.nx)])
plt.xlabel("y")
plt.ylabel("u")
plt.show()

# y-middle slice v_velocity
iy_mid = int(np.floor(v.shape[1]/2))
x_slice = prob.X_v[:,iy_mid]
y_slice = prob.Y_v[:,iy_mid]
v_slice = v[:,iy_mid]
v_slice_interp = interp(x_slice,y_slice)[:,1]
plt.plot(x_slice,v_slice)
plt.plot(x_slice,v_slice_interp,'x')
plt.plot(x_slice,v_slice)
plt.plot(x_slice,v_slice_interp,'x')
plt.legend(["Ghia et al 1982","N="+str(prob.nx)])
plt.xlabel("x")
plt.ylabel("v")
plt.show()