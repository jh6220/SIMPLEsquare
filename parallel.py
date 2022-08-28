import numpy as np
from multiprocessing import Pool
import solverFunctions as sf
import os

def SolveRandom(i):
    # Solves random BC problem and if converges saves the solution to a file, if does not coverge saves the boundary condition file
    converged,u,v,p,prob,_ = sf.SolveCFD(30,1,i=i+1,order_bc = 2,continuous=False,alpha_uv = 0.5,alpha_p = 0.2,dif_tolerance=10**(-4),max_iterations=3000)
    if converged:
        np.save(r"DataDC\Raw\dc_u_30_1_2_"+str(i)+".npy",u)
        np.save(r"DataDC\Raw\dc_v_30_1_2_"+str(i)+".npy",v)
        np.save(r"DataDC\Raw\dc_p_30_1_2_"+str(i)+".npy",p)
    else:
        converged,u,v,p,_,_ = sf.SolveCFDfromBC(prob.BCu,prob.BCv,i,alpha_uv=0.05,alpha_p=0.02,order_grid=1,rho=1,mu=0.01,dt=np.inf,CDS=True,dif_tolerance=10**(-4),max_iterations=10000)
        if converged:
            np.save(r"DataDC\Raw\dc_u_30_1_2_"+str(i)+".npy",u)
            np.save(r"DataDC\Raw\dc_v_30_1_2_"+str(i)+".npy",v)
            np.save(r"DataDC\Raw\dc_p_30_1_2_"+str(i)+".npy",p)
        else:
            np.save(r"DataDC\Raw\dc_BCu_30_1_2_"+str(i)+".npy",prob.BCu)
            np.save(r"DataDC\Raw\dc_BCv_30_1_2_"+str(i)+".npy",prob.BCv)

def SolveFromBCfile(i):
    # Solves problem from a boundary condition file, if converged saves the solution to a file and removes the BC file
    try:
        BCu_slice = np.load(r"DataDC\Raw\dc_BCu_30_1_5_"+str(i)+".npy")
        BCv_slice = np.load(r"DataDC\Raw\dc_BCv_30_1_5_"+str(i)+".npy")
        converged,u,v,p,prob,log = sf.SolveCFDfromBC(BCu_slice,BCv_slice,i,alpha_uv=0.05,alpha_p=0.02,order_grid=1,rho=1,mu=0.01,dt=np.inf,CDS=True,dif_tolerance=10**(-4),max_iterations=10000)
        if converged:
            np.save(r"DataDC\Raw\dc_u_30_1_5_"+str(i)+".npy",u)
            np.save(r"DataDC\Raw\dc_v_30_1_5_"+str(i)+".npy",v)
            np.save(r"DataDC\Raw\dc_p_30_1_5_"+str(i)+".npy",p)
            os.remove(r"DataDC\Raw\dc_BCu_30_1_5_"+str(i)+".npy")
            os.remove(r"DataDC\Raw\dc_BCv_30_1_5_"+str(i)+".npy")
        else:
            converged,u,v,p,prob,log = sf.SolveCFDfromBC(BCu_slice,BCv_slice,i,alpha_uv=0.03,alpha_p=0.01,order_grid=1,rho=1,mu=0.01,dt=0.1,CDS=True,dif_tolerance=10**(-4),max_iterations=20000)
            if converged:
                np.save(r"DataDC\Raw\dc_u_30_1_5_"+str(i)+".npy",u)
                np.save(r"DataDC\Raw\dc_v_30_1_5_"+str(i)+".npy",v)
                np.save(r"DataDC\Raw\dc_p_30_1_5_"+str(i)+".npy",p)
                os.remove(r"DataDC\Raw\dc_BCu_30_1_5_"+str(i)+".npy")
                os.remove(r"DataDC\Raw\dc_BCv_30_1_5_"+str(i)+".npy")
            else:
                pass
    except:
        print("i={} not found".format(i))

if __name__ == '__main__':
    with Pool() as p:
        p.map(SolveRandom,list(range(4000)))