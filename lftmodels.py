
def lft_trajz(t, H0, L0, a, k, z, d, baseline_ALT, baseline_heps):
    import numpy as np
    """lft_trajz(t, H0, L0, a, k, z, d, baseline_ALT, baseline_heps):
    
    ODE MODEL: 
    eqns = {H'[t] == a - k*H[t],
       L'[t] == z*k*H[t] - d*L[t]};
       H - hepatocytes
       L - liver function test (ALT/AST)
       params:
           a| basal production rate of new hepatocytes (const here)
           k| rate of injury / death of hepatocytes (const)
           z| proportionality coefficient since number of hepatocytes != unit ALT activity in the blood. z = ALT unit activity / per dead hepatocyte
           d| rate of clearance of ALT in the blood
        
        Initial values:
        C_h=927*107*(10**6)*10 # estimate from literature of number of heps in a human liver
        C_l= roughly the initial value of the ALT trajectory at peak
    
    """    
    return  -((a*(-np.exp(d*t) + np.exp(k*t))*z)/(np.exp(d*t)*(d - k))) + (a*(-(np.exp(d*t)/d) + np.exp(k*t)/k)*k*z)/(np.exp(d*t)*(d - k)) - (np.exp(-(d*t) - k*t)*(-np.exp(d*t) + np.exp(k*t))*k*z*(H0-(a/k)))/(d - k) + (L0-(a*z)/d)/np.exp(d*t)

def lft_trajzh(t, H0, a, k):
    import numpy as np
    """lft_trajzh(t, H0, a, k):
    
    ODE MODEL (HEPATOCYTE ONLY): 
    eqns = {H'[t] == a - k*H[t],
       H - hepatocytes
       params:
           a| basal production rate of new hepatocytes (const here)
           k| rate of injury / death of hepatocytes (const)
        
        Initial values:
        C_h=927*107*(10**6)*10 # estimate from literature of number of heps in a human liver
    
    """    
    return (a/k) + np.exp(-k*t)*(H0 - (a/k))

def lft_traj(t, L0, H0, a, d, k):
    """lft_traj(t, C_h, C_l, a, d, k):
    
    ODE MODEL: 
    eqns = {H'[t] == a - k*H[t],
       L'[t] == k*H[t] - d*L[t]};
       H - hepatocytes
       L - liver function test (ALT/AST)
       params:
           a| basal production rate of new hepatocytes (const here)
           k| rate of injury / death of hepatocytes (first order)
           d| rate of clearance of ALT in the blood
        
        Initial values:
        C_h=927*107*(10**6)*10 # estimate from literature of number of heps in a human liver
        C_l= roughly the initial value of the ALT trajectory at peak
    
    """    
    import numpy as np

    
    return (a*(d*np.exp(d*t)*(-1 + np.exp(k*t)) - np.exp(k*t)*(-1 + np.exp(d*t))*k) + d*(np.exp(d*t)*H0*k - np.exp(k*t)*(H0*k - d*L0 + k*L0)))/(d*np.exp((d + k)*t)*(d - k))

def lft_traj_base(t, H0, L0, a, d, bLFT, bHEP):
    """lft_traj_base(t, H0, L0, a, d, bl_lft, bl_heps)
    #    
    
    ODE MODEL: 
    eqns = {H'[t] == a - k*H[t],
       L'[t] == z*k*H[t] - d*L[t]};
       H - hepatocytes
       L - liver function test (ALT/AST)
       params:
           a| basal production rate of new hepatocytes (const here)
           k| rate of injury / death of hepatocytes (const)
           z| proportionality coefficient since number of hepatocytes != unit ALT activity in the blood. z = ALT unit activity / per dead hepatocyte
           d| rate of clearance of ALT in the blood
           L0| initial LFT
           H0| initial # hepatoctes
       **  solution is then re-parameterized in terms of baseline (ie, steady state) LFT (bl_lft) or hepatocytes (bl_heps) as follows ** 
       1. Eliminate z using z=bl_lft*d/a
       2. Eliminate k using k=a/bl_heps

    """    
    import numpy as np
    bl_lft = bLFT
    bl_heps = bHEP
    
    return (-bl_lft + L0)/np.exp(d*t) - (bl_lft*d*(-np.exp(d*t) + np.exp((a*t)/bl_heps)))/(np.exp(d*t)*(d - a/bl_heps)) - (bl_lft*d*np.exp(-(d*t) - (a*t)/bl_heps)*(-np.exp(d*t) + np.exp((a*t)/bl_heps))*(H0 - bl_heps))/((d - a/bl_heps)*bl_heps) + (a*bl_lft*d*(-(np.exp(d*t)/d) + (np.exp((a*t)/bl_heps)*bl_heps)/a))/(np.exp(d*t)*(d - a/bl_heps)*bl_heps)

def lft_traj_baseh(t, H0, a, bl_heps):
    import numpy as np
    
    return (H0 - bl_heps)/np.exp((a*t)/bl_heps) + bl_heps

def coupled_AST_ALT(t, t_ast_start, H0, L0, L0a, a, d1, d2, bl_lft, bl_lfta, bl_heps):
    import numpy as np
    t_alt=t[0:t_ast_start]
    t_ast=t[t_ast_start:]
        
    x_alt=lft_traj_base(t_alt, H0, L0, a, d1, bl_lft, bl_heps)
    x_ast=lft_traj_base(t_ast, H0, L0a, a, d2, bl_lfta, bl_heps)
    
    out = np.append(x_alt,x_ast)
    
    return out

def ALT_no_heps(t, L0, C, d):
    import numpy as np
    
    return C/d + np.exp(-d*t)*(-(C/d) + L0)

def ALT_no_heps_forced(t, L0, C0, dh, d):
    import numpy as np
    
    C=C0*np.exp(-dh*t)
    
    return C/d + np.exp(-d*t)*(-(C/d) + L0)

def biexponential(t, C, L0, a, d1, d2):
    import numpy as np
    
    return C + L0*a*np.exp(-d1*t)+L0*(1-a)*np.exp(-d2*t)

def UnitStep(v): 
    import numpy as np
    
    if type(v) is np.ndarray:
        out=np.array([])

        for ele in v:
            if ele >= 0: 
                out=np.append(out,1)
            else: 
                out=np.append(out,0)
    else:
        if v >= 0: 
            out=1
        else: 
            out=0

    return out

def square_impulse(t, C, L0, k, tau, d):
    import numpy as np
    
    return (-C - k + np.exp(d*t)*(C + k) + d*L0 + (np.exp(tau*d) - np.exp(d*t))*k*UnitStep(-tau + t))/(d*np.exp(d*t))

def proportional_death(t, C, L0, d):
    import numpy as np
    
    return (1 + (-1 + d*L0)/np.exp(C*d*t))/d

def single_param(t, C, L0):
    import numpy as np
    
    slope=1.602
    intercept=4.079
    d= np.exp((np.log(C)-intercept)/slope)
    
    return C/d + np.exp(-d*t)*(-(C/d) + L0)
    
