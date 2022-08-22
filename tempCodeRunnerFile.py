u,v = np.zeros((prob.nx+1,prob.ny+2)),np.zeros((prob.nx+2,prob.ny+1)) # if reshaped/flattened aixs=1 (y) will remain together
# u_star,v_star = np.zeros((prob.nx+1,prob.ny+2)),np.zeros((prob.nx+2,prob.ny+1)) # intermediate step in the calculation
# u_vector,v_vector = u[1:-1,1:-1].flatten(),v[1:-1,1:-1].flatten()
# p,p_prime,p_vector = np.zeros((prob.nx,prob.ny)),np.zeros((prob.nx,prob.ny)),np.zeros(prob.nx*prob.ny)