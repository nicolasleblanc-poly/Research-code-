
# TBD 
# ITEM is another option to the BFGS solver 

function item(x0, Dual, validityfunc, mineigfunc,tsolver, ei, ei_tr, chi_invdag, Gdag, Pv)
    mu=10e-8
    L=10e6
    q=mu/L
    Ak=0
    xk=x0
    yk=x0
    zk=x0
    # Where is the association with the problem at hand?
    # Where is |T> in here?
    tol = 1e-5
    cdn = false
    grad = zeros(len(x0))
    fSlist=[]
    indom = false
    while cdn == false && indom == false # setup for 20 steps
        Ak=((1+q)*Ak+2*(1+sqrt((1+Ak)*(1+q*Ak))))/(1-q)^2
        bk=Ak/((1-q)*Ak)
        dk=((1-q^2)*Ak-(1+q)*Ak)/(2*(1+q+q*Ak))
        # store the previous xk, yk and zk values to calculate the norm
        xk_m1=xk 
        yk_m1=yk
        zk_m1=zk
        # Let's calculate the new yk 
        yk=(1-bk)*zk_m1+bk*xk_m1
        # We need to solve for |T> with the yk multipliers
        val_yk = Dual(yk, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, get_grad=True)
        T_yk = val_yk[0] # we don't use this here but it was used to calculate the gradient below
        g_yk = val_yk[1] # this is the gradient evaluated at the |T> found with the yk multipliers
        # We can now calculate the new xk and zk values
        xk=yk-(1/L)*g_yk*yk_m1
        zk=(1-q*dk)*zk_m1+q*dk*yk_m1-(dk/L)*g_yk
        # Check if it is still necessary to go to the next iteration by
        # verifying the tolerance and if the smallest eigenvalue is positive,
        # which indicates we are in the domain. 
        if norm(xk-xk_m1)<tol && validityfunc(yk, chi_invdag, Gdag, Pv)>0
        cdn = true
        indom = true
        end 
        if np.linalg.norm(yk-yk_m1)<tol && validityfunc(xk, chi_invdag, Gdag, Pv)>0
        cdn = true
        indom = true
        end 
        if norm(zk-zk_m1)<tol && validityfunc(zk, chi_invdag, Gdag, Pv)>0
        cdn = True
        indom = True  
        end 
    end 
    val_yk = Dual(yk, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, get_grad=True)
    D_yk = val_yk[0]
    g_yk = val_yk[1]
    obj_yk = val_yk[2]
    val_xk = Dual(yk, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, get_grad=True)
    D_xk = val_xk[0]
    g_xk = val_xk[1]
    obj_xk = val_xk[2]
    val_zk = Dual(yk, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, get_grad=True)
    D_zk = val_zk[0]
    g_zk = val_zk[1]
    obj_zk = val_zk[2]
    dof = [yk, xk, zk]
    grad = [g_yk, g_xk, g_zk]
    dualval =[D_yk, D_xk, D_zk]
    objval = [obj_yk, obj_xk, obj_zk]
    print("results for yk, results for xk, results for zk")
    return dof, grad, dualval, objval
end 
