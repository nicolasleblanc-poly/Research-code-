module bicgstab
export bicgstab_matrix 
using LinearAlgebra
# Based on the example code from p. 686 (or p.696 of the pdf) of the 
# Introduction to Numerical Analysis textbook. 
# Code for when using a matrix and not an operator.  
# This is a biconjugate gradient program without a preconditioner. 
# m is the maximum number of iterations
function bicgstab_matrix(A,b)
    v_m1 = p_m1 = xk_m1 = zeros(ComplexF64,length(b),1)
    tol = 1e-4

    # Ax=0 since the initial xk is 0
    r0 = r_m1 = b 
    rho_m1 = alpha = omega_m1 = 1
    # for k in 1:length(b)
    for k in 1:1000
        rho_k = conj.(transpose(r0))*r_m1
        
        # beta calculation
        # First term 
        first_term = rho_k/rho_m1
        # Second term 
        second_term = alpha/omega_m1
        # Calculation 
        beta = first_term*second_term
        
        pk = r_m1 + beta.*(p_m1-omega_m1.*v_m1)
        
        vk = A*pk
        # (l[1])*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, pk)

        # alpha calculation
        # Bottom term
        bottom_term = conj.(transpose(r0))*vk
        # Calculation 
        alpha = rho_k/bottom_term 
        
        h = xk_m1 + alpha.*pk
        # If h is accurate enough, then set xk=h and quantity
        # What does accurate enough mean?

        s = r_m1 - alpha.*vk

        t = A*s
        # (l[1])*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, s)

        # omega_k calculation 
        # Top term 
        ts = conj.(transpose(t))*s
        # Bottom term
        tt = conj.(transpose(t))*t
        # Calculation 
        omega_k = ts/tt

        xk_m1 = h + omega_k.*s

        r_old = r_m1

        r_m1 = s-omega_k.*t

        # print("conj.(transpose(r_m1))*r_m1  ", conj.(transpose(r_m1))*r_m1 , "\n")
        if real((conj.(transpose(r_m1))*r_m1)[1]) < tol
        # if norm(r_m1)-norm(r_old) < tol
            print("bicgstab break \n")
            print("real((conj.(transpose(r_m1))*r_m1)[1])",real((conj.(transpose(r_m1))*r_m1)[1]),"\n")
            break 
        end 

    end
    return xk_m1
end

A = Array{Float64}(undef, 2, 2)
A[1,1] = 4
A[1,2] = 1
A[2,1] = 1
A[2,2] = 3 

b = Array{Float64}(undef, 2, 1)
b[1,1] = 1
b[2,1] = 2

print("Test ", bicgstab_matrix(A,b), "\n")

end