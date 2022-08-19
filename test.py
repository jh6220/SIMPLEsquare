# import postProcessingFunctions as ppf

# itt = 1000-1
# # ppf.PlotFlow(u_log[itt],v_log[itt],prob)

# u_n = (u_log[itt,:,:-1]+u_log[itt,:,1:])/2
# v_n = (v_log[itt,:-1,:]+v_log[itt,1:,:])/2

# plt.pcolormesh(prob.X_n,prob.Y_n,u_n)
# plt.colorbar()
# plt.show()

# plt.pcolormesh(prob.X_n,prob.Y_n,v_n)
# plt.colorbar()
# plt.show()


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
# 
# # plt.pcolormesh(prob.X_u,prob.Y_u,field[:,:,1])
# # plt.colorbar()
# # plt.show()
# 
# 
# u_res,v_res = ppf.CalcMomentumResiduals(u_true,v_true,p_true,prob)
# con_res = ppf.CalcContinuityResiduals(u_true,v_true,prob)
# print(u_res)
# print(v_res)
# print(con_res)
# # ppf.PlotFlow(u_true,v_true,prob)
# 
# mp_e = prob.rho*prob.dy_p*u_true[1:,1:-1]
# mp_w = prob.rho*prob.dy_p*u_true[:-1,1:-1]
# mp_n = prob.rho*prob.dx_p*v_true[1:-1,1:]
# mp_s = prob.rho*prob.dx_p*v_true[1:-1,:-1]
# dmp = mp_w - mp_e + mp_s - mp_n
# con_res = np.square(dmp).mean()
# print(con_res)
# =============================================================================

ax = plt.subplot()
ax.scatter(prob.X_n,prob.Y_n)
ax.set_aspect('equal', adjustable='box')
plt.xlabel("x")
plt.ylabel("y")


# =============================================================================
# def get1Dgrid(n,order=1):
#     # returns a distribution
#     y = np.linspace(0,1,n)
#     for i in range(order):
#         y = (-np.cos(y*np.pi)+1)/2
#     return y
# 
# plt.plot(get1Dgrid(20,0))
# plt.plot(get1Dgrid(20,1))
# plt.plot(get1Dgrid(20,2))
# plt.plot(get1Dgrid(20,3))
# plt.legend()
# =============================================================================

