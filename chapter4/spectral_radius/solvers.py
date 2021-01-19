import numpy as np
from scipy.sparse import diags
from scipy.sparse import bmat
from scipy.sparse.linalg import gmres
import scipy.sparse.linalg as spla
import scipy as sp
import time

# =============================================================================
# Analytical solution
# =============================================================================

def analytical(phi_init,C_init,v,SigmaA,nuSigmaF,beta,lam,S0,t):
    A = np.zeros([2,2])
    A[0,0] = v*((1.0-beta)*nuSigmaF- SigmaA)
    A[0,1] = v*lam
    A[1,0] = beta*nuSigmaF
    A[1,1] = -lam
    
    [w,vec] = np.linalg.eig(A)
    vec_inv = np.linalg.inv(vec)
    
    # Independent source
    S = np.array([[v*S0],[0]])
    Q = vec_inv.dot(S)
    
    # Initial condition
    Phi0 = np.array([[phi_init],[C_init]])
    u0   = vec_inv.dot(Phi0)
    
    # Normal vector solution
    def u(i):
        if w[i] == 0.0:
            return u0[i] + Q[i]*t
        else:
            return u0[i]*np.e**(w[i]*t) + Q[i]/w[i]*(np.e**(w[i]*t) - 1.0)
    u1 = u(0)
    u2 = u(1)
    
    # Actual solution
    N = len(t)
    phi = np.zeros(N)
    C = np.zeros(N)
    for n in range(N):
        ux = np.array([[u1[n]],[u2[n]]])
        [phi[n], C[n]] = vec.dot(ux)
    return phi,C

def BE_infinite(phi_init,C_init,v,SigmaA_list,nuSigmaF,beta,lam,S_list,dt,N):
    # Allocate solution
    phi = np.zeros(N+1)
    C = np.zeros(N+1)
    
    # Matrices
    I    = np.zeros([2,2]); I[0,0] = 1.0; I[1,1] = 1.0
    A    = np.zeros([2,2])

    # Initial condition
    phi[0] = phi_init
    C[0]   = C_init
    Phi    = np.array([[phi[0]],[C[0]]])
    
    for n in range(0,N):
        SigmaA = SigmaA_list
        S      = S_list
        A[0,0] = v*(SigmaA-(1-beta)*nuSigmaF)
        A[0,1] = -v*lam
        A[1,0] = -beta*nuSigmaF
        A[1,1] = lam
        M  = I/dt + A
        M  = np.linalg.inv(M)
        Q    = np.array([[v*S],[0]])
        Phi  = M.dot(Phi/dt + Q)
        phi[n+1] = Phi[0]
        C[n+1]   = Phi[1]
    return phi, C

def MB_infinite(phi_init,C_init,v,SigmaA_list,nuSigmaF,beta,lam,S_list,dt,N):
    # Allocate solution
    phi = np.zeros(N+1)
    C = np.zeros(N+1)
    
    # Matrices
    I    = np.zeros([2,2]); I[0,0] = 1.0; I[1,1] = 1.0
    A    = np.zeros([2,2])

    # Initial condition
    phi[0] = phi_init
    C[0]   = C_init
    Phi    = np.array([[phi[0]],[C[0]]])
    
    for n in range(0,N):
        SigmaA = SigmaA_list[n]
        S      = S_list[n]
        A[0,0] = v*(SigmaA-(1-beta)*nuSigmaF)
        A[0,1] = -v*lam
        A[1,0] = -beta*nuSigmaF
        A[1,1] = lam
        M  = I/dt + (dt/2*A).dot(A+2*I/dt)
        M  = np.linalg.inv(M)
        Q    = np.array([[v*S],[0]])
        Phi  = M.dot(Phi/dt + (I+dt/2*A).dot(Q))
        phi[n+1] = Phi[0]
        C[n+1]   = Phi[1]
    return phi, C

# =============================================================================
# Numerical solution
# =============================================================================

def BE(phi_init,C_init,v,SigmaA,nuSigmaF,D,beta,lam,Q,dt,N,X,J,aR,aL,tol):
    # Time!
    time_start = time.time()

    # =========================================================================
    # Create A
    # =========================================================================

    # Misc.
    dx     = X/J
    dx_sq  = dx*dx
    
    # nuSigmaF
    nuSigmaFd = beta*nuSigmaF
    nuSigmaFp = nuSigmaF - nuSigmaFd
    
    # SigmaJ
    SigmaJ = np.zeros([J+1])
    SigmaJ[0] = 2*D[0]/dx_sq  
    for j in range(1,J):
        SigmaJ[j] = 2/(1.0/D[j-1] + 1.0/D[j])/dx_sq
    SigmaJ[J] = 2*D[-1]/dx_sq
    
    # Delayed contribution
    ld  = lam*dt
    exp = np.e**-ld
    xi0 = (1-exp)/ld - exp
    xi1 = 1 - xi0 - exp
    xiC = exp
    
    # BC
    BL = 1.0/((1-aL)/(1+aL)*dx/D[0]/4+1)
    BR = 1.0/((1-aR)/(1+aR)*dx/D[-1]/4+1)
    
    # Upper and lower diagonal
    A_ul = -SigmaJ[1:-1]
    # Diagonal
    A_d      = SigmaA - (nuSigmaFp + xi1*nuSigmaFd) + 1/v/dt + SigmaJ[:-1] + SigmaJ[1:]
    A_d[0]  -= SigmaJ[0]*BL
    A_d[-1] -= SigmaJ[-1]*BR
    # The A
    A = diags([A_ul,A_d,A_ul],[-1,0,1],format="csc")

    # Create Preconditioner
    Pr = spla.spilu(A)
    Pr = spla.LinearOperator(A.shape, Pr.solve)

    # =========================================================================
    # Preparation
    # =========================================================================    
    
    # Allocate solution
    phi   = np.zeros(J)
    C     = np.zeros(J)
    b     = np.zeros(J)
    phi_t = np.zeros([N,J])
    C_t   = np.zeros([N,J])
    
    # Initial condition
    phi[:] = phi_init[:]
    C[:]   = C_init[:]

    # =========================================================================
    # Solve
    # =========================================================================
    
    # March in time        
    for n in range(0,N):
        # Set RHS
        b[:] = Q + (1./v/dt+xi0*nuSigmaFd)*phi + xiC*lam*C
        # Solve C (partially)
        C = xiC*C + xi0*nuSigmaFd/lam*phi
        # Solve phi
        phi[:], exitCode = gmres(A, b, x0=phi, M=Pr, tol=tol*1E-2)
        # Solve C
        C += xi1*nuSigmaFd/lam*phi
        # Store solutions
        phi_t[n][:] = phi[:]
        C_t[n][:]   = C[:]

    # Time!
    time_end = time.time()  
    return phi_t, C_t, time_end - time_start

def MB(phi_init,C_init,v,SigmaA,nuSigmaF,D,beta,lam,Q,dt,N,X,J,aR,aL,tol):
    # Time!
    time_start = time.time()

    # =========================================================================
    # Create A
    # =========================================================================
    
    # Misc.
    dx     = X/J
    dx_sq = dx*dx
    
    # nuSigmaF
    nuSigmaFd = beta*nuSigmaF
    nuSigmaFp = nuSigmaF - nuSigmaFd
    
    # SigmaJ
    SigmaJ = np.zeros([J+1])
    SigmaJ[0] = 2*D[0]/dx_sq  
    for j in range(1,J):
        SigmaJ[j] = 2/(1.0/D[j-1] + 1.0/D[j])/dx_sq
    SigmaJ[J] = 2*D[-1]/dx_sq
    
    # Delayed contribution
    ld      = lam*dt
    denom   = 1/(1+ld+ld**2/2)
    xi12_1  = ld*denom
    xi1_1   = ld**2/2*denom
    xi12_12 = 1-1*denom
    xi1_12  = -ld/2*denom
    xiC_1   = denom
    xiC_12  = (1-denom)/ld
  
    # BC
    BL = 1.0/((1-aL)/(1+aL)*dx/D[0]/4+1)
    BR = 1.0/((1-aR)/(1+aR)*dx/D[-1]/4+1)
    
    # Block A1
    A_ul     = -SigmaJ[1:-1]
    A_d      = SigmaA - (nuSigmaFp+xi12_12*nuSigmaFd) + SigmaJ[:-1] + SigmaJ[1:]
    A_d[0]  -= SigmaJ[0]*BL
    A_d[-1] -= SigmaJ[-1]*BR
    A1 = diags([A_ul,A_d,A_ul],[-1,0,1],format="csc")
    
    # Block A4
    A_d      = SigmaA - (nuSigmaFp+xi1_1*nuSigmaFd) + 2./v/dt + SigmaJ[:-1] + SigmaJ[1:]
    A_d[0]  -= SigmaJ[0]*BL
    A_d[-1] -= SigmaJ[-1]*BR
    A4 = diags([A_ul,A_d,A_ul],[-1,0,1],format="csc")
    
    # Block A2
    A_d = 1/(v*dt) - xi1_12*nuSigmaFd
    A2  = diags([A_d],[0],format="csc")
    
    # Block A3
    A_d = -2/(v*dt) - xi12_1*nuSigmaFd
    A3  = diags([A_d],[0],format="csc")
      
    # Create A
    A = bmat([[A1,A2],[A3,A4]],format="csc")
    
    # Create Pr
    Pr = spla.spilu(A)
    Pr = spla.LinearOperator(A.shape, Pr.solve)

    # =========================================================================
    # Preparation
    # =========================================================================    
    
    # Allocate solution
    b     = np.zeros(2*J)
    phi   = np.zeros(2*J)
    C     = np.zeros(J)
    phi_t = np.zeros([N,J])
    C_t   = np.zeros([N,J])
    
    # Initial condition
    phi[:J] = phi_init[:]
    phi[J:] = phi_init[:]
    C[:]   = C_init[:]

    # =========================================================================
    # Solve
    # =========================================================================    
    
    # March in time        
    for n in range(0,N):
        # Set RHS
        b[:J] = Q + phi[J:]/v/dt + xiC_12*lam*C
        b[J:] = Q + xiC_1*lam*C
        # Solve phi
        phi[:], exitCode = gmres(A, b, x0=phi, M=Pr, tol=tol*1E-2)
        # Solve C
        C = xiC_1*C + nuSigmaFd/lam*(xi12_1*phi[:J]+xi1_1*phi[J:])
        # Store solutions
        phi_t[n][:] = phi[J:]
        C_t[n][:]   = C[:]

    # Time!
    time_end = time.time()    
    return phi_t, C_t, time_end - time_start

# =============================================================================
# Numerical solution Transport
# =============================================================================

def BE_trans(psi_init,C_init,v,SigmaT,nuSigmaF,SigmaS,beta,lam,Q,dt,K,N,X,J,eps,accelerate,moc):
        
    # Time!
    time_start = time.time()

    # Misc.
    dx = X/J
    
    # XS
    nuSigmaFd = beta*nuSigmaF
    nuSigmaFp = nuSigmaF - nuSigmaFd
    SigmaA    = SigmaT - SigmaS
       
    # Delayed contribution
    ld  = lam*dt
    exp = np.e**-ld
    xi0 = (1-exp)/ld - exp
    xi1 = 1 - xi0 - exp
    xiC = exp
    
    # Quadratures
    mu, w = np.polynomial.legendre.leggauss(N)
     
    # =========================================================================
    # Preparation (General)
    # =========================================================================    
    
    # Allocate solution
    phi     = np.zeros(J)
    phi_old = np.zeros(J)
    psi_avg = np.copy(psi_init)
    C       = np.zeros(J)
    phi_t   = np.zeros([K,J])
    C_t     = np.zeros([K,J])
    rho_t = []
    
    # Containers
    S     = np.zeros(J)    
    
    # Initial condition
    for n in range(N):
        for j in range(J):
            phi[j] = phi[j] + psi_avg[n,j]*w[n]
            
    anal = BE_infinite(phi[0],C_init[0],v,SigmaA[0],nuSigmaF[0],beta,lam,Q[0],dt,K)
    err_t = []
    
    C[:] = C_init[:]
    
    # Counter
    N_iter = 0

    # =========================================================================
    # Preparation (Transport sweeps)
    # =========================================================================
    
    # Sweeps
    def sweep_sndd(S,phi,psi_avg,n,start,end,increment,update=False,psi_in=0.0):
        # Sweep
        for j in range(start,end,increment):
            # Constants
            tau  = SigmaT[j]*dx/abs(mu[n])
            eta  = v*SigmaT[j]*dt
            zeta = 1 + 1.0/eta
            C1   = 1/tau
            C2   = 0.5*zeta

            # psi
            psi_out = ((C1 - C2)*psi_in + (S[j] + psi_avg[n,j]/eta))/(C1 + C2)
            psi_bar = 0.5*(psi_in + psi_out)

            # phi
            phi[j] = phi[j] + psi_bar*w[n]

            # update?
            if update: psi_avg[n,j] = psi_bar
            
            # reset
            psi_in = psi_out
        return psi_out
    
    def sweep_moc(S,phi,psi_avg,n,start,end,increment,update=False,psi_in=0.0):
        # Sweep
        for j in range(start,end,increment):
            # constants
            tau  = SigmaT[j]*dx/abs(mu[n])
            eta  = v*SigmaT[j]*dt
            zeta = 1 + 1.0/eta 
            Ss = (S[j] + psi_avg[n,j]/eta)/zeta
            C  = tau*zeta

            # psi
            exp = np.e**(-tau*zeta)
            psi_out = (psi_in - Ss)*exp + Ss
            psi_bar = Ss - (psi_out-psi_in)/C

            # phi
            phi[j]  = phi[j] + psi_bar*w[n]

            # update?
            if update: psi_avg[n,j] = psi_bar
            
            # reset
            psi_in  = psi_out
        return psi_out

    # Pick sweep
    if moc: sweep = sweep_moc
    else:   sweep = sweep_sndd

    def run_sweep(phi_old,S,phi,psi_avg,update=False):
        # Reset phi
        phi_old[:] = phi[:]
        phi[:] = 0.0

        # Set source
        for j in range(J):
            xi = (SigmaS[j]+nuSigmaF[j])/SigmaT[j]
            S[j] = 0.5*(xi*phi_old[j] + Q[j]/SigmaT[j])

        # Sweeps
        for n in range(int(N/2),N):
            # Forward sweep
            psi_out = sweep(S,phi,psi_avg,n,0,J,1,update)
            # Backward sweep
            sweep(S,phi,psi_avg,N-1-n,J-1,-1,-1,update,psi_out)

    # =========================================================================
    # Preparation (DSA)
    # =========================================================================
    
    # For DSA
    if accelerate:
        # Misc.
        dx     = X/J
        dx_sq  = dx*dx
        F = np.zeros(J)
        D    = 1.0/(3*(SigmaT + 1.0/v/dt))
        aL   = 0.0
        aR   = 1.0
        
        # SigmaJ
        SigmaJ = np.zeros([J+1])
        SigmaJ[0] = 2*D[0]/dx_sq  
        for j in range(1,J):
            SigmaJ[j] = 2/(1.0/D[j-1] + 1.0/D[j])/dx_sq
        SigmaJ[J] = 2*D[-1]/dx_sq
        
        # BC
        BL = 1.0/((1-aL)/(1+aL)*dx/D[0]/4+1)
        BR = 1.0/((1-aR)/(1+aR)*dx/D[-1]/4+1)
        
        # Upper and lower diagonal
        A_ul = -SigmaJ[1:-1]
        # Diagonal
        A_d      = SigmaA - (nuSigmaFp + xi1*nuSigmaFd) + 1/v/dt + SigmaJ[:-1] + SigmaJ[1:]
        A_d[0]  -= SigmaJ[0]*BL
        A_d[-1] -= SigmaJ[-1]*BR
        # The A
        A_dsa = diags([A_ul,A_d,A_ul],[-1,0,1],format="csc")
    
        # Create Preconditioner
        P_dsa = spla.spilu(A_dsa)
        P_dsa = spla.LinearOperator(A_dsa.shape, P_dsa.solve)
    
    def dsa(phi,phi_old,F,Q,tol):
        # Set error/correction source
        Q[:] = (phi-phi_old)*(SigmaS+nuSigmaF)

        # Solve
        F[:], exitCode = gmres(A_dsa, Q, M=P_dsa, tol=tol)
        if exitCode:
            print("GMRES not converged")
            exit()

        # Update
        for j in range(J):
            phi[j] = phi[j] + F[j]

    # =========================================================================
    # Solve
    # =========================================================================
    
    # March in time
    for k in range(K):
        # Initial errors and spectral radius
        err = 1.0+eps
        rho = 1.0
        rho_denom = 1.0
        tol = eps

        # SI starts
        while err > tol:
        #while N_iter<10:
            # Sweep!
            run_sweep(phi_old,S,phi,psi_avg)
            N_iter = N_iter + 1

            # DSA
            if accelerate:
                if tol < 0.0:
                    toli = eps*1E-2
                else:
                    toli = tol*1E-2
                dsa(phi,phi_old,F,S,toli)

            # Error
            phi_old[:] = phi - phi_old # phi_old stores error
            diff_norm  = np.linalg.norm(phi_old)
            err        = np.linalg.norm(abs(np.true_divide(phi_old,phi)))
            
            # Spectral radius
            rho       = diff_norm/rho_denom
            rho_denom = diff_norm
            rho_t.append(rho)
            err_t.append(phi[-1])

            # Tolerance considering false convergence
            tol = (1-rho)*eps
            print("be",err,rho)

        # TODO !!!!!!!!!!!!!            
        # Final sweep to update psi_avg
        run_sweep(phi_old,S,phi,psi_avg,True)
        N_iter = N_iter + 1
        
        # Store
        phi_t[k][:] = phi[:]

    # Time!
    time_end = time.time()
    return phi_t, C_t, psi_avg, N_iter, time_end - time_start, rho_t, err_t, anal[0][0]

def MB_trans(psi_init,C_init,v,SigmaT,nuSigmaF,SigmaS,beta,lam,Q,dt,K,N,X,J,eps,accelerate,moc,store):
    # Time!
    time_start = time.time()

    # Misc.
    dx = X/J
    
    # XS
    nuSigmaFd = beta*nuSigmaF
    nuSigmaFp = nuSigmaF - nuSigmaFd
    SigmaA    = SigmaT - SigmaS
    
    # Delayed contribution
    ld      = lam*dt
    denom   = 1/(1+ld+ld**2/2)
    xi12_1  = ld*denom
    xi1_1   = ld**2/2*denom
    xi12_12 = 1-1*denom
    xi1_12  = -ld/2*denom
    xiC_1   = denom
    xiC_12  = (1-denom)/ld
    
    # Quadratures
    mu, w = np.polynomial.legendre.leggauss(N)
    
    # =========================================================================
    # Preparation (General)
    # =========================================================================    

    # MOC + storage
    if store:
        ecos = np.zeros([N,J])
        esin = np.zeros([N,J])
        for n in range(N):
            for j in range(J):
                tau     = SigmaT[j]*dx/abs(mu[n])
                eta     = v*SigmaT[j]*dt
                trigarg = tau/eta
                exp     = np.exp(-(1+1/eta)*tau)
                ecos[n,j] = exp*np.cos(trigarg)
                esin[n,j] = exp*np.sin(trigarg)
        
    # Allocate solution
    phi1     = np.zeros(J)
    phi2     = np.zeros(J)
    phi1_old = np.zeros(J)
    phi2_old = np.zeros(J)
    psi_avg = np.copy(psi_init)
    C       = np.zeros(J)
    phi_t   = np.zeros([K,J])
    C_t     = np.zeros([K,J])
    rho_t = []

    # Containers
    S  = np.zeros(2*J)
    S1 = S[:J]
    S2 = S[J:]
    
    # Containers
    S  = np.zeros(2*J)
    S1 = S[:J]
    S2 = S[J:]

    # Initial condition
    for n in range(N):
        for j in range(J):
            phi1[j] = phi1[j] + psi_avg[n,j]*w[n]
            phi2[j] = phi2[j] + psi_avg[n,j]*w[n]
    C[:] = C_init[:]
    
    # Counter
    N_iter = 0
    
    # =========================================================================
    # Preparation (Transport sweeps)
    # =========================================================================
    
    def sweep_sndd(S1,S2,phi1,phi2,psi_avg,n,start,end,increment,update=False,psi1_in=0.0,psi2_in=0.0):
        # Sweep
        for j in range(start,end,increment):
            # constants
            tau  = SigmaT[j]*dx/abs(mu[n])
            eta  = v*SigmaT[j]*dt
            zeta = 1 + 2.0/eta
            C1   = 1/tau
            C2   = 0.5*zeta
            C3   = 0.5/eta
            C4   = 0.5*eta
            C5   = 2/tau
            C6   = C4*(C5+1)

            # psi2
            psi2_out = ((C6*(C1-C2)-C3)*psi2_in + C5*psi1_in \
                        + S1[j] + psi_avg[n,j]/eta + C6*S2[j])/(C6*(C1+C2)+C3)
            psi2_bar = 0.5*(psi2_in + psi2_out)

            # psi1
            psi1_bar = C4*(C1*(psi2_out-psi2_in) + zeta*psi2_bar - S2[j])
            psi1_out = 2*psi1_bar - psi1_in

            # phi
            phi1[j]  = phi1[j] + psi1_bar*w[n]
            phi2[j]  = phi2[j] + psi2_bar*w[n]

            # update?
            if update: psi_avg[n,j] = psi2_bar
            
            # reset
            psi1_in  = psi1_out
            psi2_in  = psi2_out
        return psi1_out, psi2_out
   
    def sweep_moc(S1,S2,phi1,phi2,psi_avg,n,start,end,increment,update=False,psi1_in=0.0,psi2_in=0.0):
        # Sweep
        for j in range(start,end,increment):
            # Constants
            eta         = v*SigmaT[j]*dt
            zeta        = 1 + 2.0/eta
            eta_inv     = 1.0/eta
            two_eta_inv = 2.0*eta_inv
            tau         = SigmaT[j]*dx/abs(mu[n])
            if store:
                A = ecos[n,j]
                B = esin[n,j]
            else:
                trigarg = tau*eta_inv
                exp     = np.e**(-(1+eta_inv)*tau)
                A       = exp*np.cos(trigarg)
                B       = exp*np.sin(trigarg)

            # A_inv
            denom = 1.0/(zeta + two_eta_inv*eta_inv)
            A_inv11 = denom*zeta
            A_inv12 = denom*-eta_inv
            A_inv21 = denom*two_eta_inv
            A_inv22 = denom

            # psip
            Qs1 = S1[j] + eta_inv*psi_avg[n,j]
            Qs2 = S2[j]
            psi1p = A_inv11*Qs1 + A_inv12*Qs2
            psi2p = A_inv21*Qs1 + A_inv22*Qs2

            # psi_out
            C1 = psi1_in - psi1p
            C2 = C1 + psi2p - psi2_in
            psi1_out = C1*A + C2*B + psi1p
            psi2_out = C1*(A+B) + C2*(B-A) + psi2p

            # psi_bar
            dpsi1   = psi1_out - psi1_in
            dpsi2   = psi2_out - psi2_in
            psi1_bar = psi1p - (A_inv11*dpsi1+A_inv12*dpsi2)/tau
            psi2_bar = psi2p - (A_inv21*dpsi1+A_inv22*dpsi2)/tau

            # phi
            phi1[j]  = phi1[j] + psi1_bar*w[n]
            phi2[j]  = phi2[j] + psi2_bar*w[n]

            # update?
            if update: psi_avg[n,j] = psi2_bar

            # reset
            psi1_in = psi1_out
            psi2_in = psi2_out
        return psi1_out, psi2_out

    # Pick sweep
    if moc: sweep = sweep_moc
    else:   sweep = sweep_sndd
    
    def run_sweep(phi1_old,phi2_old,S1,S2,phi1,phi2,psi_avg,update=False):
        # Reset phi
        phi1_old[:] = phi1[:]
        phi2_old[:] = phi2[:]
        phi1[:] = 0.0
        phi2[:] = 0.0

        # Set source
        for j in range(J):
            xi = (SigmaS[j]+nuSigmaF[j])/SigmaT[j]
            S1[j] = 0.5*(xi*phi1_old[j] + Q[j]/SigmaT[j])
            S2[j] = 0.5*(xi*phi2_old[j] + Q[j]/SigmaT[j])

        # Sweeps
        for n in range(int(N/2),N):
            # Forward sweep
            psi1_out, psi2_out = sweep(S1,S2,phi1,phi2,psi_avg,n,0,J,1,update)
            # Backward sweep
            sweep(S1,S2,phi1,phi2,psi_avg,N-1-n,J-1,-1,-1,update,psi1_out,psi2_out)

    # =========================================================================
    # Preparation (DSA)
    # =========================================================================
    
    # For DSA
    if accelerate:
        # Misc.
        dx     = X/J
        dx_sq  = dx*dx
        F  = np.ones(2*J)
        eta    = v*SigmaT*dt
        D_star = 1.0/(3*np.multiply((eta+2+2/eta),SigmaT))
        D_hat  = np.multiply(eta,D_star)
        D_tild = np.multiply((eta+2),D_star)
        aL   = 0.0
        aR   = 1.0
               
        # =====================================================================
        # A1
        # =====================================================================
 
        D = D_tild
        
        # BC
        BL = 1.0/((1-aL)/(1+aL)*dx/D[0]/4+1)
        BR = 1.0/((1-aR)/(1+aR)*dx/D[-1]/4+1)
        
        # SigmaJ
        SigmaJ = np.zeros([J+1])
        SigmaJ[0] = 2*D[0]/dx_sq  
        for j in range(1,J):
            SigmaJ[j] = 2/(1.0/D[j-1] + 1.0/D[j])/dx_sq
        SigmaJ[J] = 2*D[-1]/dx_sq 
        
        # Build
        A_ul     = -SigmaJ[1:-1]
        A_d      = SigmaA - (nuSigmaFp+xi12_12*nuSigmaFd) + SigmaJ[:-1] + SigmaJ[1:]
        A_d[0]  -= SigmaJ[0]*BL
        A_d[-1] -= SigmaJ[-1]*BR
        A1 = diags([A_ul,A_d,A_ul],[-1,0,1],format="csc")
        
        # =====================================================================
        # A4
        # =====================================================================
 
        D = D_hat
        
        # BC
        BL = 1.0/((1-aL)/(1+aL)*dx/D[0]/4+1)
        BR = 1.0/((1-aR)/(1+aR)*dx/D[-1]/4+1)
        
        # SigmaJ
        SigmaJ[0] = 2*D[0]/dx_sq  
        for j in range(1,J):
            SigmaJ[j] = 2/(1.0/D[j-1] + 1.0/D[j])/dx_sq
        SigmaJ[J] = 2*D[-1]/dx_sq         
        
        # Build
        A_ul     = -SigmaJ[1:-1]
        A_d      = SigmaA - (nuSigmaFp+xi1_1*nuSigmaFd) + 2./v/dt + SigmaJ[:-1] + SigmaJ[1:]
        A_d[0]  -= SigmaJ[0]*BL
        A_d[-1] -= SigmaJ[-1]*BR
        A4 = diags([A_ul,A_d,A_ul],[-1,0,1],format="csc")

        # =====================================================================
        # A2
        # =====================================================================
 
        D = -D_star
        
        # BC
        BR = 1.0/((1-aR)/(1+aR)*dx/D[-1]/4+1)
        
        # SigmaJ
        SigmaJ = np.zeros([J+1])
        SigmaJ[0] = 2*D[0]/dx_sq  
        for j in range(1,J):
            SigmaJ[j] = 2/(1.0/D[j-1] + 1.0/D[j])/dx_sq
        SigmaJ[J] = 2*D[-1]/dx_sq                 
        
        # Build
        A_ul     = -SigmaJ[1:-1]
        A_d      = 1/(v*dt) - xi1_12*nuSigmaFd + SigmaJ[:-1] + SigmaJ[1:]
        A_d[0]  -= SigmaJ[0]*BR
        A_d[-1] -= SigmaJ[-1]*BR
        A2 = diags([A_ul,A_d,A_ul],[-1,0,1],format="csc")
        
        # =====================================================================
        # A3
        # =====================================================================
 
        D = 2*D_star
        
        # BC
        BR = 1.0/((1-aR)/(1+aR)*dx/D[-1]/4+1)
        
        # SigmaJ
        SigmaJ = np.zeros([J+1])
        SigmaJ[0] = 2*D[0]/dx_sq  
        for j in range(1,J):
            SigmaJ[j] = 2/(1.0/D[j-1] + 1.0/D[j])/dx_sq
        SigmaJ[J] = 2*D[-1]/dx_sq                 
        
        # Build
        A_ul     = -SigmaJ[1:-1]
        A_d      = -2/(v*dt) - xi12_1*nuSigmaFd + SigmaJ[:-1] + SigmaJ[1:]
        A_d[0]  -= SigmaJ[0]*BR
        A_d[-1] -= SigmaJ[-1]*BR
        A3 = diags([A_ul,A_d,A_ul],[-1,0,1],format="csc")
    
        # Create A
        A_dsa = bmat([[A1,A2],[A3,A4]],format="csc")
        
        # Create Pr
        P_dsa = spla.spilu(A_dsa)
        P_dsa = spla.LinearOperator(A_dsa.shape, P_dsa.solve)    
        
    def dsa(phi1,phi2,phi1_old,phi2_old,F,Q,tol):
        # Set error/correction source
        Q[:J] = (phi1 - phi1_old)*(SigmaS+nuSigmaF)
        Q[J:] = (phi2 - phi2_old)*(SigmaS+nuSigmaF)

        # Solve
        F[:], exitCode = gmres(A_dsa, Q, M=P_dsa, tol=tol)
        if exitCode:
            print("GMRES not converged")
            exit()
        
        # Update
        for j in range(J):
            phi1[j] = phi1[j] + F[j]
            phi2[j] = phi2[j] + F[j+J]

    # =========================================================================
    # Solve
    # =========================================================================    
    
    # March in time
    for k in range(K):
        # Initial errors and spectral radius
        err = 1.0+eps
        rho = 1.0
        rho_denom = 1.0
        tol = eps

        # SI starts
        while err > tol:
        #while N_iter<10:
            # Sweep!
            run_sweep(phi1_old,phi2_old,S1,S2,phi1,phi2,psi_avg)
            N_iter = N_iter + 1

            # DSA
            if accelerate:
                if tol < 0.0:
                    toli = eps*1E-2
                else:
                    toli = tol*1E-2
                dsa(phi1,phi2,phi1_old,phi2_old,F,S,toli)

            # Error
            phi1_old[:] = phi1 - phi1_old # phi_old stores error
            phi2_old[:] = phi2 - phi2_old
            diff1_norm = np.linalg.norm(phi1_old)
            diff2_norm = np.linalg.norm(phi2_old)
            diff_norm  = np.sqrt(diff1_norm*diff1_norm + diff2_norm*diff2_norm)
            err1       = np.linalg.norm(abs(np.true_divide(phi1_old,phi1)))
            err2       = np.linalg.norm(abs(np.true_divide(phi2_old,phi2)))
            err        = np.sqrt(err1*err1 + err2*err2)
            
            # Spectral radius
            rho        = diff_norm/rho_denom
            rho_denom  = diff_norm
            rho_t.append(rho)
            
            # Tolerance considering false convergence
            tol = (1-rho)*eps
            print("mb",err,rho)

        # TODO !!!!!!!!!!!!!            
        # Final sweep to update psi_avg
        run_sweep(phi1_old,phi2_old,S1,S2,phi1,phi2,psi_avg,True)
        N_iter = N_iter + 1
        
        # Store
        phi_t[k][:] = phi2[:]

    # Time!
    time_end = time.time()    
    return phi_t, C_t, psi_avg, N_iter, time_end - time_start, rho_t