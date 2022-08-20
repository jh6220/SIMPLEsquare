
# =============================================================================
# import pandas as pd
# df = pd.read_csv(r"C:\Users\jakub\OneDrive\Documents\JupyterNotebook\UROP\DataCFD\ldc2d-re100-all\all\5000NUcav.plt",delim_whitespace=True)
# data = df.to_numpy()
# points = df.iloc[5000:5000+2601,0:2].to_numpy()
# df = pd.read_csv(r"C:\Users\jakub\OneDrive\Documents\JupyterNotebook\UROP\DataCFD\ldc2d-re100-all\all\5000NUcav.var",delim_whitespace=True,header=None)
# vals = df.to_numpy()
# vals
# interp = LinearNDInterpolator(points, vals)
# field = interp(prob.X_v,prob.Y_v)
# u_true = field[:,:,0]
# field = interp(prob.X_u,prob.Y_u)
# v_true = field[:,:,1]
# field = interp(prob.X_p,prob.Y_p)
# p_true = field[:,:,2]
# u_true = np.transpose(u_true)
# v_true = np.transpose(v_true)
# p_true = np.transpose(p_true)
# u_true,v_true = sf.ApplyVelBC(u_true,v_true,prob)
# =============================================================================

# =============================================================================
# def GenerateHarmonics(t,order,start=1):
#     t_in = (t[0:-1]+t[1:])/2
#     t = np.concatenate([t_in,t+1,t_in+2,t+3])*2*np.pi/4
#     # Generate a random function of given order for boundary definition
#     n = t.shape
#     x = np.zeros(n)
#     for i in range(start,order):
#         x[:] += np.random.randn()*np.cos(float(i)*t)\
#                    +(np.random.randn()*np.sin(float(i)*t))
#     return x
# 
# 
# x_in = (x[0:-1]+x[1:])/2
# x_concat = np.concatenate([x_in,x+1,x_in+2,x+3])/4
# 
# # boundary condition
# # axis=1 is the bc values along the wall; axis=0 differentiates the walls:
# # 0 - south; 1 - west; 2 - north; 3 - east
# nx = x.size-1
# BCu = np.zeros((4,nx+2))
# BCv = np.zeros((4,nx+2))
# 
# 
# # u and v is defined continuous counter clock-wise around the boundary (south->east->north->west)
# order = 3
# u = GenerateHarmonics(x,order,start=0)
# v = GenerateHarmonics(x,order,start=0)
# 
# # define the wall lengths of the velocit control volumes that are pumping mass in/out of domain
# dy = x[1:]-x[:-1]
# 
# # defines mapping between the clock-wise u and v and the system of storing boundary values with BCu/BCv
# BCu[0,1:-2] = u[0:(nx-1)]
# BCu[1,1:-1] = u[3*nx-2:4*nx-2][::-1]
# BCu[2,1:-2] = u[2*nx-1:3*nx-2][::-1]
# BCu[3,1:-1] = u[nx-1:2*nx-1]
# BCv[0,1:-1] = v[0:nx]
# BCv[1,1:-2] = v[3*nx-1:4*nx-2][::-1]
# BCv[2,1:-1] = v[2*nx-1:3*nx-1][::-1]
# BCv[3,1:-2] = v[nx:2*nx-1]
# 
# # net massflow into the domain that needs to be corrected for to satisfy continuity
# dm_dot = (dy*BCv[0,1:-1]).sum() + (dy*BCu[1,1:-1]).sum() - (dy*BCv[2,1:-1]).sum() + (dy*BCu[3,1:-1]).sum()
# BCv[0,1:-1] -= dm_dot/(2)
# BCu[1,1:-1] -= dm_dot/(2)
# BCv[2,1:-1] += dm_dot/(2)
# BCu[3,1:-1] += dm_dot/(2)
# 
# 
# =============================================================================

mp_e = prob.rho*prob.dy_p*u[1:,1:-1]
mp_w = prob.rho*prob.dy_p*u[:-1,1:-1]
mp_n = prob.rho*prob.dx_p*v[1:-1,1:]
mp_s = prob.rho*prob.dx_p*v[1:-1,:-1]
dmp = mp_w - mp_e + mp_s - mp_n
plt.imshow(dmp)