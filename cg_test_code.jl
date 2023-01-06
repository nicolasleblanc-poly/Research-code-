using LinearAlgebra
function conj_grad(A,b)
    tol = 1e-5 # The program terminates once 
    # there is an r for which its norm is smaller
    # than the chosen tolerance. 

    xk = zeros(ComplexF64,length(b),1)
    # Ax=0 since the initial xk is 0
    pk = rk = b 
    # k = 0
    for k in 1:length(b)
        # alpha_k coefficient calculation 
        # Top term
        rkrk = conj.(transpose(rk))*rk
        # Bottom term 
        A_pk = A*pk
        pk_A_pk = conj.(transpose(pk))*A_pk
        # Division
        alpha_k = rkrk/pk_A_pk

        # x_{k+1} calculation 
        xk = xk + alpha_k.*pk

        # r_{k+1} calculation 
        rk = rk - alpha_k.*A_pk

        # print("norm(rk_plus1) ",norm(rk), "\n")
        # if norm(rk)<tol
        #     return xk
        # end

        # beta_k coefficient calculation 
        # Top term 
        rkplus1_rkplus1 = conj.(transpose(rk))*rk
        # The bottom term is the same one calculated earlier 
        # Division 
        beta_k = rkplus1_rkplus1/rkrk

        pk = rk + beta_k.*pk

        # global k+=1 

    end
    return xk
end 


A = zeros(Int8, 2,2)
A[1,1] = 4
A[1,2] = 1
A[2,1] = 1
A[2,2] = 3

b = zeros(Int8, 2,1)
b[1,1]=1
b[2,1]=2

# x_julia = cg(A, b)
print("x  ", conj_grad(A,b))