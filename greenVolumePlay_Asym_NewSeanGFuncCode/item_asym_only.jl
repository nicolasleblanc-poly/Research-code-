# ITEM is another option to the BFGS solver 
module item_asym_only 
using LinearAlgebra, bfgs_power_iteration_asym_only
export ITEM 

function ITEM(gMemSlfN, gMemSlfA,x0,Dual,P,chi_inv_coeff,ei,cellsA,validityfunc)
    mu=10e-8
    L=10e6
    q=mu/L
    Ak=0
    xk=x0
    yk=x0
    zk=x0
    # Where is the association with the problem at hand?
    # Where is |T> in here?
    # For both of my question, I'm guessing the gradient is the only important quantity
    # here since it relates to the problem that is being considered. 
    tol = 1e-5 # Tolerance (error between current and previous multiplier value)
    cdn = false
    grad = zeros(Float64, length(x0), 1)  # zeros(len(x0))
    fSlist=[]
    indom = false
    while cdn == false && indom == false # Setup for 20 steps
        Ak=((1+q)*Ak+2*(1+sqrt((1+Ak)*(1+q*Ak))))/(1-q)^2
        bk=Ak/((1-q)*Ak)
        dk=((1-q^2)*Ak-(1+q)*Ak)/(2*(1+q+q*Ak))
        # Store the previous xk, yk and zk values to calculate the norm
        xk_m1=xk # The m1 in xk_m1 means minus 1, so it is the previous term to xk
        yk_m1=yk
        zk_m1=zk
        # Let's calculate the new yk 
        yk=(1-bk)*zk_m1+bk*xk_m1
        # We need to solve for |T> with the yk multipliers
        val_yk = Dual(yk,grad,P,ei,gMemSlfN,gMemSlfA, chi_inv_coeff, cellsA,[],true)
        # Old: (yk, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, get_grad=true)
        T_yk = val_yk[1] # We don't use this here but it was used to calculate the gradient below
        g_yk = val_yk[2] # This is the gradient evaluated at the |T> found with the yk multipliers
        # We can now calculate the new xk and zk values
        xk=yk-(1/L)*g_yk*yk_m1
        zk=(1-q*dk)*zk_m1+q*dk*yk_m1-(dk/L)*g_yk
        # Check if it is still necessary to go to the next iteration by
        # verifying the tolerance and if the smallest eigenvalue is positive,
        # which indicates we are in the domain. 
        if norm(xk-xk_m1)<tol && validityfunc(xk,cellsA,gMemSlfN, gMemSlfA, chi_inv_coeff, P)>0
        cdn = true
        indom = true
        end 
        if norm(yk-yk_m1)<tol && validityfunc(yk,cellsA,gMemSlfN, gMemSlfA, chi_inv_coeff, P)>0
        cdn = true
        indom = true
        end 
        if norm(zk-zk_m1)<tol && validityfunc(zk,cellsA,gMemSlfN, gMemSlfA, chi_inv_coeff, P)>0
        cdn = true
        indom = true  
        end 
    end 
    # yk 
    val_yk = Dual(yk,grad,P,ei,gMemSlfN,gMemSlfA, chi_inv_coeff, cellsA,[],true)
    # (yk, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, get_grad=true)
    D_yk = val_yk[1]
    g_yk = val_yk[2]
    obj_yk = val_yk[3]

    # xk 
    val_xk = Dual(xk,grad,P,ei,gMemSlfN,gMemSlfA, chi_inv_coeff, cellsA,[],true)
    # Dual(yk, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, get_grad=true)
    D_xk = val_xk[1]
    g_xk = val_xk[2]
    obj_xk = val_xk[3]

    # zk 
    val_zk = Dual(zk,grad,P,ei,gMemSlfN,gMemSlfA, chi_inv_coeff, cellsA,[],true)
    # (yk, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, get_grad=true)
    D_zk = val_zk[1]
    g_zk = val_zk[2]
    obj_zk = val_zk[3]

    # Different return values 
    dof = [yk, xk, zk]
    grad = [g_yk, g_xk, g_zk]
    dualval =[D_yk, D_xk, D_zk]
    objval = [obj_yk, obj_xk, obj_zk]
    print("results for yk, results for xk, results for zk")
    return dof, grad, dualval, objval
end 
end 