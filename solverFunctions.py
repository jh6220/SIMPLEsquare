# =============================================================================
# Functions for defining a CFD solver with rectangular control and SIMPLE algorithm
#
# Note on indexing convention:
# - in 2D arrays the x and y coordinate are defined as: [i_x,i_y]
# - u-velocity is defined in positive x index direction
# - v-velocity is defined in positive y index direction
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve, bicgstab
from scipy.interpolate import LinearNDInterpolator
import copy
import time

def get1Dgrid(n,order=1):
    # returns a distribution
    y = np.linspace(0,1,n)
    for i in range(order):
        y = (-np.cos(y*np.pi)+1)/2
    return y

class CFDproblem():
    # Structure that stores constant of a CFD problem:
    #   - control volume positions
    #   - control volume wall lengths
    #   - boundary conditions
    #   - fluid propeties: density, viscosity
    #   - solver setting: time step (if infinity than SIMPLE algorithm), under-relaxation factor, 
    #       CDS (central differencing scheme) correction
    #   - viscous shear coefficients
    
    def __init__(self,x,y,rho,mu,BCu,BCv,dt=np.inf,alpha_uv=0.07,alpha_p=0.02,CDS=True):
        # definition for control volumes centroids (p-pressurel; u,v-momentum)
        self.xc_u = x
        self.yc_v = y
        self.xc_p = (self.xc_u[:-1]+self.xc_u[1:])/2
        self.xc_v = np.concatenate([[-self.xc_p[0]],self.xc_p,[1+self.xc_p[0]]]) # adding the "ghost" velocities beyond the boundaries
        self.yc_p = (self.yc_v[:-1]+self.yc_v[1:])/2
        self.yc_u = np.concatenate([[-self.yc_p[0]],self.yc_p,[1+self.xc_p[0]]]) # adding the "ghost" velocities beyond the boundaries
        # meshgrid of p-CV corners
        self.Y_n,self.X_n = np.meshgrid(self.yc_u,self.xc_v)
        # meshgrid of p-CV centroids
        self.Y_p,self.X_p = np.meshgrid(self.yc_p,self.xc_p)
        # meshgrid of u-CV centriods
        self.Y_u,self.X_u = np.meshgrid(self.yc_u,self.xc_u)
        # meshgrid of v-CB centroids
        self.Y_v,self.X_v = np.meshgrid(self.yc_v,self.xc_v)
        # definition for control volumes vertexes (p-pressurel; u,v-momentum)
        self.xv_p = self.xc_u
        self.yv_p = self.yc_v
        # number of pressure control volumes
        self.nx = len(self.xc_p)
        self.ny = len(self.yc_p) 
        # pressure control volumes wall lengths 
        self.dx_p = (self.xv_p[1:]-self.xv_p[:-1]).reshape(self.nx,1)
        self.dy_p = (self.yv_p[1:]-self.yv_p[:-1]).reshape(1,self.ny)
        # u_momentum control volumes wall lengths (only for internal CV)
        self.dx_u = (self.dx_p[1:]+self.dx_p[:-1])/2
        self.dy_u = self.dy_p
        # v_momentum control volumes wall lengths (only for internal CV) 
        self.dx_v = self.dx_p
        self.dy_v = (self.dy_p[:,1:]+self.dy_p[:,:-1])/2
        # pressure control volumes controid distances
        self.lx_p = (self.xc_p[1:]-self.xc_p[:-1]).reshape(self.nx-1,1)
        self.ly_p = (self.yc_p[1:]-self.yc_p[:-1]).reshape(1,self.ny-1)
        # u-momentum control volumes controid distances
        self.lx_u = (self.xc_u[1:]-self.xc_u[:-1]).reshape(self.nx,1)
        self.ly_u = (self.yc_u[1:]-self.yc_u[:-1]).reshape(1,self.ny+1)
        # v-momentum control volumes controid distances
        self.lx_v = (self.xc_v[1:]-self.xc_v[:-1]).reshape(self.nx+1,1)
        self.ly_v = (self.yc_v[1:]-self.yc_v[:-1]).reshape(1,self.ny)
        #boundary conditions
        self.BCu = BCu # u-momentum
        self.BCv = BCv # v-momentum
        
        self.rho = rho # density
        self.mu = mu # viscosity
        self.dt = dt # time step size
        self.alpha_uv = alpha_uv # momentum under-relaxation factor
        self.alpha_p = alpha_p # pressure under-relaxation factor
        self.CDS = CDS # bool whether to use CDS correction
        
        # viscous shear coefficients
        self.Aud_x = self.mu * self.dy_u/self.lx_u # diffusion coeffcient for u-momentum-CV in x direction from current CV
        self.Aud_y = self.mu * self.dx_u/self.ly_u # diffusion coeffcient for u-momentum-CV in y direction from current CV
        self.Avd_x = self.mu * self.dy_v/self.lx_v # diffusion coeffcient for v-momentum-CV in x direction from current CV
        self.Avd_y = self.mu * self.dx_v/self.ly_v # diffusion coeffcient for v-momentum-CV in y direction from current CV

def GenerateHarmonics(n,order,start=1):
    # Generate a random function of given order for boundary definition
    t = np.linspace(0,2*np.pi,n)
    x = np.zeros(n)
    for i in range(start,order):
        x[:] += np.random.randn()*np.cos(float(i)*t)\
                   +(np.random.randn()*np.sin(float(i)*t))
    return x

def GetBC2(nx):
    #boundary condition
    #axis=1 is the bc values along the wall; axis=0 differentiates the walls:
    #0 - south; 1 - west; 2 - north; 3 - east
    BCu = np.zeros((4,nx+2))
    BCv = np.zeros((4,nx+2))
    
    
    # u and v is defined continuous counter clock-wise around the boundary (south->east->north->west)
    order = 3
    u = GenerateHarmonics(4*nx-2,order,start=0)
    v = GenerateHarmonics(4*nx-2,order,start=0)
    
    # defines mapping between the clock-wise u and v and the system of storing boundary values with BCu/BCv
    BCu[0,1:-2] = u[0:(nx-1)]
    BCu[1,1:-1] = u[3*nx-2:4*nx-2][::-1]
    BCu[2,1:-2] = u[2*nx-1:3*nx-2][::-1]
    BCu[3,1:-1] = u[nx-1:2*nx-1]
    BCv[0,1:-1] = v[0:nx]
    BCv[1,1:-2] = v[3*nx-1:4*nx-2][::-1]
    BCv[2,1:-1] = v[2*nx-1:3*nx-1][::-1]
    BCv[3,1:-2] = v[nx:2*nx-1]
    
    # net massflow into the domain that needs to be corrected for to satisfy continuity
    dm_dot = BCv[0,1:-1].sum() + BCu[1,1:-1].sum() - (BCv[2,1:-1].sum() + BCu[3,1:-1].sum())
    BCv[0,1:-1] -= dm_dot/(4*nx)
    BCu[1,1:-1] -= dm_dot/(4*nx)
    BCv[2,1:-1] += dm_dot/(4*nx)
    BCu[3,1:-1] += dm_dot/(4*nx)
    return BCu,BCv,u,v

def ApplyVelBC(u,v,prob):
    # applyies the boundary condition the 2d-array of velocities
    # most important for recalculating the ghost-velocities after every velocity update
    
    u[:,0] = 2*prob.BCu[0,:u.shape[0]] - u[:,1]
    u[0,:] = prob.BCu[1,:u.shape[1]]
    u[:,-1] = 2*prob.BCu[2,:u.shape[0]] - u[:,-2]
    u[-1,:] = prob.BCu[3,:u.shape[1]]
    
    u[0,0] = (u[0,1]+u[1,0])/2
    u[0,-1] = (u[0,-2]+u[1,-1])/2
    u[-1,0] = (u[-1,1]+u[-2,0])/2
    u[-1,-1] = (u[-1,-2]+u[-2,-1])/2
    
    v[:,0] = prob.BCv[0,:v.shape[0]]
    v[0,:] = 2*prob.BCv[1,:v.shape[1]] - v[1,:]
    v[:,-1] = prob.BCv[2,:v.shape[0]]
    v[-1,:] = 2*prob.BCv[3,:v.shape[1]] - v[-2,:]
    
    v[0,0] = (v[0,1]+v[1,0])/2
    v[0,-1] = (v[0,-2]+v[1,-1])/2
    v[-1,0] = (v[-1,1]+v[-2,0])/2
    v[-1,-1] = (v[-1,-2]+v[-2,-1])/2
    return u,v

def MomentumEqU(u,v,p,prob):
    # defines the linear u-momentum linear system of equation (Au * u_vector = bu)
    # Au_p stores the diagonal values of the matrix Au which is used in the pressure correction
    
    # initialize the arrays
    Au=sparse.lil_matrix(((prob.nx-1)*(prob.ny),(prob.nx-1)*(prob.ny)))
    Au_p = np.zeros((prob.nx-1,prob.ny))
    bu=np.zeros(((prob.nx-1)*(prob.ny),1))
    
    
    #mu_ ~ mass flow rate through cv wall
    #indexes in internal CV matrix system
    mu_e = prob.rho*prob.dy_u*(u[1:-1,1:-1]+u[2:,1:-1])/2
    mu_w = prob.rho*prob.dy_u*(u[:-2,1:-1]+u[1:-1,1:-1])/2
    mu_n = prob.rho*prob.dx_u*(v[1:-2,1:]+v[2:-1,1:])/2
    mu_s = prob.rho*prob.dx_u*(v[1:-2,:-1]+v[2:-1,:-1])/2
    
    #indexes in internal u-CV system; i_all = i_internal+1
    for ix in range(0,prob.nx-1):
        for iy in range(0,prob.ny):
            # defines the index if the matrix
            iA = (prob.ny)*(ix) + (iy)
            Au_e=Au_w=Au_n=Au_s=0
            # defines the coefficients for the difference equations for neighbouring velocities
            #east
            Au_e = -max(-mu_e[ix,iy],0) - prob.Aud_x[ix+1,iy]
            if (ix==prob.nx-2):
                bu[iA] -= Au_e*u[ix+2,iy+1]
            else:
                Au[iA,iA+(prob.ny)] = Au_e
            #west
            Au_w = -max(mu_w[ix,iy],0) - prob.Aud_x[ix,iy]
            if (ix==0):
                bu[iA] -= Au_w*u[ix,iy+1]
            else:
                Au[iA,iA-(prob.ny)] = Au_w
            #north
            Au_n = -max(-mu_n[ix,iy],0) - prob.Aud_y[ix,iy+1]
            if (iy==prob.ny-1):
                bu[iA] -= Au_n*u[ix+1,iy+2]
            else:
                Au[iA,iA+1] = Au_n
            #south
            Au_s = -max(mu_s[ix,iy],0) - prob.Aud_y[ix,iy]
            if (iy==0):
                bu[iA] -= Au_s*u[ix+1,iy]
            else:
                Au[iA,iA-1] = Au_s
            #cell centre
            Au[iA,iA] = Au_p[ix,iy] = prob.rho*prob.dx_u[ix,0]*prob.dy_u[0,iy]/prob.dt-\
            (Au_e+Au_w+Au_n+Au_s)+\
            (mu_e[ix,iy]-mu_w[ix,iy])+(mu_n[ix,iy]-mu_s[ix,iy])
            #source-term
            #pressure term + unsteady term
            bu[iA] += prob.dy_u[0,iy]*(p[ix,iy]-p[ix+1,iy]) + prob.rho*prob.dx_u[ix,0]*prob.dy_u[0,iy]*u[ix+1,iy+1]/prob.dt
            
            if prob.CDS:
            # central-difference-scheme correction ( + F_UDS_(m-1) - F_CDS_(m-1)) to the source term
            # this term will apprach 0 in convergence
                #-> forcing term with UDS from last time step (F_UDS_(m-1)) 
                bu[iA] += -max(-mu_e[ix,iy],0)*u[ix+2,iy+1]-max(mu_w[ix,iy],0)*u[ix,iy+1]+\
                -max(-mu_n[ix,iy],0)*u[ix+1,iy+2]-max(mu_s[ix,iy],0)*u[ix+1,iy]+\
                +(max(mu_e[ix,iy],0)+max(-mu_w[ix,iy],0)+max(mu_n[ix,iy],0)+max(-mu_s[ix,iy],0))*u[ix+1,iy+1] ####
                #-> forcing term with CDS from last time step (F_CDS_(m-1)) 
                bu[iA] -= (mu_e[ix,iy]*(u[ix+2,iy+1]+u[ix+1,iy+1])/2\
                + mu_n[ix,iy]*(u[ix+1,iy+2]+u[ix+1,iy+1])/2\
                - mu_w[ix,iy]*(u[ix,iy+1]+u[ix+1,iy+1])/2\
                - mu_s[ix,iy]*(u[ix+1,iy]+u[ix+1,iy+1])/2)
            
    Au = Au.tocsr()
    return Au,bu,Au_p
            
def MomentumEqV(u,v,p,prob):
    # defines the linear v-momentum linear system of equation (Av * v_vector = bv)
    # Av_p stores the diagonal values of the matrix Au which is used in the pressure correction
    Av=sparse.lil_matrix(((prob.nx)*(prob.ny-1),(prob.nx)*(prob.ny-1)))
    Av_p = np.zeros((prob.nx,prob.ny-1))
    bv=np.zeros(((prob.nx-1)*(prob.ny),1))
    
    #mv ~ m_dot ~ mass flow rate through cv wall
    #indexes in internal CV matrix system
    mv_e = prob.rho*prob.dy_v*(u[1:,1:-2]+u[1:,2:-1])/2
    mv_w = prob.rho*prob.dy_v*(u[:-1,1:-2]+u[:-1,2:-1])/2
    mv_n = prob.rho*prob.dx_v*(v[1:-1,1:-1]+v[1:-1,2:])/2
    mv_s = prob.rho*prob.dx_v*(v[1:-1,1:-1]+v[1:-1,:-2])/2
    
    #indexes in internal v-CV system; i_all = i_internal+1
    for ix in range(0,prob.nx):
        for iy in range(0,prob.ny-1):
            iA = (prob.ny-1)*(ix) + (iy)
            Av_e=Av_w=Av_n=Av_s=0
            #east
            Av_e = -max(-mv_e[ix,iy],0) - prob.Avd_x[ix+1,iy] 
            if (ix==prob.nx-1):
                bv[iA] -= Av_e*v[ix+2,iy+1]
            else:
                Av[iA,iA+(prob.ny-1)] = Av_e
            #west
            Av_w = -max(mv_w[ix,iy],0) - prob.Avd_x[ix,iy] 
            if (ix==0):
                bv[iA] -= Av_w*v[ix,iy+1]
            else:
                Av[iA,iA-(prob.ny-1)] = Av_w
            #north
            Av_n = -max(-mv_n[ix,iy],0) - prob.Avd_y[ix,iy+1]
            if (iy==prob.ny-2):
                bv[iA] -= Av_n*v[ix+1,iy+2]
            else:
                Av[iA,iA+1] = Av_n
            #south
            Av_s = -max(mv_s[ix,iy],0) - prob.Avd_y[ix,iy]
            if (iy==0):
                bv[iA] -= Av_s*v[ix+1,iy]
            else:
                Av[iA,iA-1] = Av_s
            #cell centre
            Av[iA,iA] = Av_p[ix,iy] = prob.rho*prob.dx_v[ix,0]*prob.dy_v[0,iy]/prob.dt-\
            (Av_e+Av_w+Av_n+Av_s)+\
            (mv_e[ix,iy]-mv_w[ix,iy])+(mv_n[ix,iy]-mv_s[ix,iy])
            
            #source-term
            #pressure term + unsteady term
            bv[iA] += prob.dx_v[ix,0]*(p[ix,iy]-p[ix,iy+1]) + prob.rho*prob.dx_v[ix,0]*prob.dy_v[0,iy]*v[ix+1,iy+1]/prob.dt
            
            if prob.CDS:
            #central-difference-scheme correction ( + F_UDS_(m-1) - F_CDS_(m-1))
                #-> forcing term with UDS from last time step (F_UDS_(m-1)) 
                bv[iA] += -max(-mv_e[ix,iy],0)*v[ix+2,iy+1]-max(mv_w[ix,iy],0)*v[ix,iy+1]+\
                -max(-mv_n[ix,iy],0)*v[ix+1,iy+2]-max(mv_s[ix,iy],0)*v[ix+1,iy]+\
                +(max(mv_e[ix,iy],0)+max(-mv_w[ix,iy],0)+max(mv_n[ix,iy],0)+max(-mv_s[ix,iy],0))*v[ix+1,iy+1] ####
                #-> forcing term with CDS from last time step (F_CDS_(m-1)) 
                bv[iA] -= (mv_e[ix,iy]*(v[ix+2,iy+1]+v[ix+1,iy+1])/2\
                + mv_n[ix,iy]*(v[ix+1,iy+2]+v[ix+1,iy+1])/2\
                - mv_w[ix,iy]*(v[ix,iy+1]+v[ix+1,iy+1])/2\
                - mv_s[ix,iy]*(v[ix+1,iy]+v[ix+1,iy+1])/2)
            
    Av = Av.tocsr()
    return Av,bv,Av_p
 
def SolveMomentum(A,b,u_star):
    # solves the momentum system of equations and updates the velocity array
    # uses scipy DIRECT linear system solver
    u_vector = spsolve(A,b)
    u_mat = u_vector.reshape(u_star.shape[0]-2,u_star.shape[1]-2)
    u_star[1:-1,1:-1] = u_mat
    return u_star,u_vector

def SolveMomentumI(A,b,u_star,u_vector):
    # solves the momentum system of equations and updates the velocity array
    # uses scipy ITERATIVE linear system solver
    u_vector = bicgstab(A,b,x0=u_vector,tol=10*-3)[0]
    u_mat = u_vector.reshape(u_star.shape[0]-2,u_star.shape[1]-2)
    u_star[1:-1,1:-1] = u_mat
    return u_star,u_vector

def pressureCorrectionCoff(u,v,p,Au_p,Av_p,prob):
    # Calculates the linear system of equation which represents the pressure correction
    # Poisson equation that needs to solved to satisfy continuity equation
    
    # Arrays initialization
    Ap = sparse.lil_matrix(((prob.nx)*(prob.ny),(prob.nx)*(prob.ny)))
    bp = np.zeros((prob.nx)*(prob.ny))
    
    #mp ~ mass flow rate through cv wall
    #indexes in internal CV matrix system
    mp_e = prob.rho*prob.dy_p*u[1:,1:-1]
    mp_w = prob.rho*prob.dy_p*u[:-1,1:-1]
    mp_n = prob.rho*prob.dx_p*v[1:-1,1:]
    mp_s = prob.rho*prob.dx_p*v[1:-1,:-1]
    # mass flow rate IN to eac CV
    dmp = mp_w - mp_e + mp_s - mp_n
    
    for ix in range(prob.nx):
        for iy in range(prob.ny):
            iA = (prob.ny)*(ix) + (iy)
            
            #east
            if (ix==prob.nx-1):
                Ap_e=0
            else:
                Ap[iA,iA+prob.ny] = Ap_e = -prob.rho*prob.dy_p[0,iy]**2/Au_p[ix,iy]
            #west
            if (ix==0):
                Ap_w=0
            else:
                Ap[iA,iA-prob.ny] = Ap_w = -prob.rho*prob.dy_p[0,iy]**2/Au_p[ix-1,iy]
            #north
            if (iy==prob.ny-1):
                Ap_n=0
            else:
                Ap[iA,iA+1] = Ap_n = -prob.rho*prob.dx_p[ix,0]**2/Av_p[ix,iy]
            #south
            if (iy==0):
                Ap_s=0
            else:
                Ap[iA,iA-1] = Ap_s = -prob.rho*prob.dx_p[ix,0]**2/Av_p[ix,iy-1]
            #cell centre
            Ap[iA,iA] = -(Ap_e+Ap_w+Ap_n+Ap_s)
            bp[iA] = -dmp[ix,iy]
            
            # in the incompressible N-V eqautions the only the pressure gradient appears
            # => at CV [0,0] the pressure value will be defined as 0 and all other pressure values
            # will be defined in respect to it
            if (ix==0 and iy==0):
                Ap[iA,iA] = 1
                bp[iA] = 0
                Ap[iA,iA+1] = 0
                Ap[iA,iA+prob.ny]=0
            
    Ap = Ap.tocsr()
    return Ap,bp,dmp

def SolvePressure(Ap,bp,prob):
    # solves the pressure-correction system of equations and updates the velocity array
    # uses scipy DIRECT linear system solver
    p_vector = spsolve(Ap,bp)
    p_prime = p_vector.reshape(prob.nx,prob.ny)
    return p_prime,p_vector

def SolvePressureI(Ap,bp,p_vector,prob):
    # solves the pressure-correction system of equations and updates the velocity array
    # uses scipy ITERATIVE linear system solver
    p_vector = bicgstab(Ap,bp,x0=p_vector,tol=10*-3)[0]
    p_prime = p_vector.reshape(prob.nx,prob.ny)
    return p_prime,p_vector

def VelocityCorrection(p_prime,Au_p,Av_p,u,v,prob):
    # calculates the velocity corrections u_prime and v_prime from the pressure-correction
    # p_prime
    u_prime = np.zeros((u.shape[0]-2,u.shape[1]-2))
    v_prime = np.zeros((v.shape[0]-2,v.shape[1]-2))
    
    # u-velocity correction
    for ix in range(u_prime.shape[0]):
        for iy in range(u_prime.shape[1]):
            u_prime[ix,iy] = -prob.dy_p[0,iy]*(p_prime[ix+1,iy]-p_prime[ix,iy])/Au_p[ix,iy]
            
    # v-velocity correction
    for ix in range(v_prime.shape[0]):
        for iy in range(v_prime.shape[1]):
            v_prime[ix,iy] = -prob.dx_p[ix,0]*(p_prime[ix,iy+1]-p_prime[ix,iy])/Av_p[ix,iy]
    
    return u_prime,v_prime