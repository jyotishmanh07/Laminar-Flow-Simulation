import numpy as np
import shutil
import os

def ddx(f, dx):
    """
    Calculate first derivative with respect to x using central difference
    Args:
        f: 2D array of values
        dx: grid spacing in x direction
    Returns:
        2D array of df/dx values
    """
    return (f[1:-1, 2:] - f[1:-1, 0:-2]) / (2 * dx)

def ddy(f, dy):
    """
    Calculate first derivative with respect to y using central difference
    Args:
        f: 2D array of values
        dy: grid spacing in y direction
    Returns:
        2D array of df/dy values
    """
    return (f[2:, 1:-1] - f[0:-2, 1:-1]) / (2 * dy)

def upwind_ddx(f, dx):
    """First-order upwind differencing for x-derivative"""
    return (f[1:-1, 1:-1] - f[1:-1, :-2]) / dx

def upwind_ddy(f, dy):
    """First-order upwind differencing for y-derivative"""
    return (f[1:-1, 1:-1] - f[:-2, 1:-1]) / dy

def ddx2(f, dx):
    """
    Calculate second derivative with respect to x using central difference
    Args:
        f: 2D array of values
        dx: grid spacing in x direction
    Returns:
        2D array of d²f/dx² values
    """
    return (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, 0:-2]) / (dx**2)

def ddy2(f, dy):
    """
    Calculate second derivative with respect to y using central difference
    Args:
        f: 2D array of values
        dy: grid spacing in y direction
    Returns:
        2D array of d²f/dy² values
    """
    return (f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[0:-2, 1:-1]) / (dy**2)

def pressure_poisson(p, dx, dy, rho, dt, u, v, pBCs):
    pn = np.empty_like(p)
    pn = p.copy()
    nit = 50   #pseudo-time steps in each timestep
    
    for q in range(nit):
        pn = p.copy()
        
        # Calculate derivatives
        du_dx = ddx(u, dx)
        dv_dy = ddy(v, dy)
        du_dy = ddy(u, dy)
        dv_dx = ddx(v, dx)
        
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          (rho * (1 / dt * (du_dx + dv_dy) -
                                 du_dx**2 - 2 * du_dy * dv_dx - dv_dy**2)))

        pressureBC(pBCs[0], pBCs[1], pBCs[2], pBCs[3], p, dx, dy)
    
    return p

def set_time_step(u, v, dx, dy, Re, tau=0.5):
    """
    Calculate timestep based on combined CFL condition including both
    convective and viscous stability requirements.
    Args:
        u: x-velocity field
        v: y-velocity field
        dx: grid spacing in x
        dy: grid spacing in y
        nu: kinematic viscosity
        tau: safety factor (default 0.5)
    Returns:
        dt: timestep size
    """
    # Handle zero velocities
    u_max = max(np.max(np.abs(u)), 1e-10)
    v_max = max(np.max(np.abs(v)), 1e-10)
    
    # Convective limit
    #dt_conv = min(dx/u_max, dy/v_max)
    
    # Viscous limit
    dt_visc = (Re /2) * (1/dx**2 + 1/dy**2)**(-1)
    
    # Combined timestep
    dt = tau * min(dx/u_max, dy/v_max, dt_visc)
    
    return dt

def flow_solver(nt, u, v, dt, ds, dx, dy, p, rho, Re, pBCs, uBCs, vBCs):
    """
    Solve for flow field using predictor-corrector with adaptive timestepping
    """
    un = np.empty_like(u)
    vn = np.empty_like(v)
    
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        
        # Calculate adaptive timestep
        dt = set_time_step(un, vn, dx, dy, Re)
        
        print(f"Step {n}, dt: {dt:.5e}, max U: {np.max(un):.3f}, max V: {np.max(vn):.3f}")
        
        # Pressure-velocity coupling
        p = pressure_poisson(p, dx, dy, rho, dt, u, v, pBCs)
        
        """ # Update u velocity
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt * ddx(un, dx) -
                        vn[1:-1, 1:-1] * dt * ddy(un, dy) -
                        dt / (2 * rho) * ddx(p, dx) +
                        dt/Re * (ddx2(un, dx) + ddy2(un, dy)))

        # Update v velocity
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt * ddx(vn, dx) -
                        vn[1:-1, 1:-1] * dt * ddy(vn, dy) -
                        dt / (2 * rho) * ddy(p, dy) +
                        dt/Re * (ddx2(vn, dx) + ddy2(vn, dy))) """
        
        # Update u velocity with upwind scheme for convective terms
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt * upwind_ddx(un, dx) -
                        vn[1:-1, 1:-1] * dt * upwind_ddy(un, dy) -
                        dt / (2 * rho) * ddx(p, dx) +
                        dt/Re * (ddx2(un, dx) + ddy2(un, dy)))

        # Update v velocity with upwind scheme for convective terms
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt * upwind_ddx(vn, dx) -
                        vn[1:-1, 1:-1] * dt * upwind_ddy(vn, dy) -
                        dt / (2 * rho) * ddy(p, dy) +
                        dt/Re * (ddx2(vn, dx) + ddy2(vn, dy)))
        
        velmag = (u**2 + v**2)**0.5
        
        # Calculate velocity magnitude
        velmag = (u**2 + v**2)**0.5
        
        # Save results periodically
        if ((n+1) % ds == 0):
            np.savetxt(f'u{n//ds}.csv', u, delimiter=',')
            np.savetxt(f'v{n//ds}.csv', v, delimiter=',')
            np.savetxt(f'p{n//ds}.csv', p, delimiter=',')
            np.savetxt(f'Umag{n//ds}.csv', velmag, delimiter=',')

        # Apply boundary conditions
        XVelBC(uBCs[0], uBCs[1], uBCs[2], uBCs[3], u, dx, dy)
        YVelBC(vBCs[0], vBCs[1], vBCs[2], vBCs[3], v, dx, dy)
                
    return u, v, p

def pressureBC(a,b,c,d,p,dx,dy):
    #left
    if a[0]=='N':
        p[:, 0]=a[1]*dx+p[:, 1]
    elif a[0]=='D':
        p[:,0]=a[1]
    else:
        return('Please enter a valid BC type')

    #right
    if b[0]=='N':
        p[:, -1]=b[1]*dx+p[:, -2]
    elif b[0]=='D':
        p[:, -1]=b[1]
    else:
        return('Please enter a valid BC type')

    #top
    if c[0]=='N':
        p[-1,:]=c[1]*dy+p[-2, :]
    elif c[0]=='D':
        p[-1,:]=c[1]
    else:
        return('Please enter a valid BC type')

    #bottom
    if d[0]=='N':
        p[0,:]=d[1]*dy+p[1, :]
    elif d[0]=='D':
        p[0,:]=d[1]
    else:
        return('Please enter a valid BC type')

def XVelBC(a,b,c,d,u,dx,dy):
    #left
    if a[0]=='N':
        u[:, 0]=a[1]*dx+u[:, 1]
    elif a[0]=='D':
        u[:,0]=a[1]
    else:
        return('Please enter a valid BC type')

    #right
    if b[0]=='N':
        u[:, -1]=b[1]*dx+u[:, -2]
    elif b[0]=='D':
        u[:, -1]=b[1]
    else:
        return('Please enter a valid BC type')

    #top
    if c[0]=='N':
        u[-1,:]=c[1]*dy+u[-2, :]
    elif c[0]=='D':
        u[-1,:]=c[1]
    else:
        return('Please enter a valid BC type')

    #bottom
    if d[0]=='N':
        u[0,:]=d[1]*dy+u[1, :]
    elif d[0]=='D':
        u[0,:]=d[1]
    else:
        return('Please enter a valid BC type')

def YVelBC(a,b,c,d,v,dx,dy):
    #left
    if a[0]=='N':
        v[:, 0]=a[1]*dx+v[:, 1]
    elif a[0]=='D':
        v[:,0]=a[1]
    else:
        return('Please enter a valid BC type')

    #right
    if b[0]=='N':
        v[:, -1]=b[1]*dx+v[:, -2]
    elif b[0]=='D':
        v[:, -1]=b[1]
    else:
        return('Please enter a valid BC type')

    #top
    if c[0]=='N':
        v[-1,:]=c[1]*dy+v[-2, :]
    elif c[0]=='D':
        v[-1,:]=c[1]
    else:
        return('Please enter a valid BC type')

    #bottom
    if d[0]=='N':
        v[0,:]=d[1]*dy+v[1, :]
    elif d[0]=='D':
        v[0,:]=d[1]
    else:
        return('Please enter a valid BC type')


def clearResults():
    fileList=os.listdir('./')
    
    if os.path.isdir('./Results'):
        shutil.rmtree('./Results')
        os.mkdir('Results')
    else:
        os.mkdir('Results')
    
    for file in fileList:
        if file.endswith('.csv'):
            shutil.move(file,'Results/.')
