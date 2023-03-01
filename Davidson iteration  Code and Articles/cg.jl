module cg 
export cg_matrix
using LinearAlgebra
# Based on the example code from https://en.wikipedia.org/wiki/Conjugate_gradient_method
# Code for the AA case 

# m is the maximum number of iterations
function cg_matrix(A,b)
    tol = 1e-5 # The program terminates once 
    # there is an r for which its norm is smaller
    # than the chosen tolerance. 

    xk = zeros(ComplexF64,length(b),1)
    # Ax=0 since the initial xk is 0
    pk = rk = b 
    # k = 0
    # for k in 1:length(b)
    for k in 1:1000
        # alpha_k coefficient calculation 
        # Top term
        rkrk = conj.(transpose(rk))*rk
        # Bottom term 
        A_pk = A*pk
        pk_A_pk = conj.(transpose(pk))*A_pk
        # Division
        alpha_k = rkrk/pk_A_pk

        rk_old = rk

        # x_{k+1} calculation 
        xk = xk + alpha_k.*pk

        # r_{k+1} calculation 
        rk = rk - alpha_k.*A_pk

        # print("norm(rk_plus1) ",norm(rk), "\n")
        if norm(rk) - norm(rk_old) <tol
            return xk
        end

        # beta_k coefficient calculation 
        # Top term 
        rkplus1_rkplus1 = conj.(transpose(rk))*rk
        # The bottom term is the same one calculated earlier 
        # Division 
        print("rkplus1_rkplus1 ", rkplus1_rkplus1, "\n")
        print("rkrk ", rkrk, "\n")

        beta_k = rkplus1_rkplus1/rkrk

        pk = rk + beta_k.*pk

    end
    return xk
end 

A = Array{Float64}(undef, 2, 2)
A[1,1] = 4
A[1,2] = 1
A[2,1] = 1
A[2,2] = 3 

b = Array{Float64}(undef, 2, 1)
b[1,1] = 1
b[2,1] = 2

print("A ", A, "\n")
print("Test ", cg_matrix(A,b), "\n")

end 