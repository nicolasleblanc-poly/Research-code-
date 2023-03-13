module Davidson_HarmonizRitz_TestFile
using LinearAlgebra, Random, Arpack, KrylovKit, bicgstab, cg

function davidson_it(A)
    """
    This function is the Davidson iteration algorithm. It is a hopefully better
    alternative to the Power iteration method since it uses past information to 
    choose search directions and at each new iteration. We use harmonic Rizt 
    vectors here, which allows to directly find the minimum eigenvalue. 
    """
    # m = 20 # Amount of iterations of the inner loop

    # Part 1. Setup 
    tol = 1e-3 # Tolerance for which the program will converge 

    rows = size(A)[1]
    cols = size(A)[2]

    v = zeros(ComplexF64, rows, 1)
    rand!(v)
    print("v ", v, "\n")

    vk = v/norm(v) # Vector 
    wk = A*v # Vector 
    lk = (conj.(transpose(wk))*vk)[1] # Number
    hk = (conj.(transpose(wk))*wk)[1] # Number
    print("hk ", hk, "\n")

    # Vk = Array{Float64}(undef, rows, cols)
    Vk = zeros(ComplexF64, rows, cols)
    # The first index is to select a row
    # The second index is to select a column
    Vk[:,1] = vk
    print("Vk ", Vk, "\n")

    # Wk = Array{Float64}(undef, rows, cols)
    Wk = zeros(ComplexF64, rows, cols)
    Wk[:,1] = wk
    print("Wk ", Wk, "\n")
    
    Lk = zeros(ComplexF64, rows, cols)
    # Lk = Array{Float64}(undef, rows, cols)
    Lk[1,1] = lk
    print("Lk ", Lk, "\n")

    Hk_hat = zeros(ComplexF64, rows, cols)
    # Hk_hat = Array{Float64}(undef, rows, cols)
    Hk_hat[1,1] = hk
    print("Hk_hat ", Hk_hat, "\n")

    Hk_tilde = zeros(ComplexF64, rows, cols)

    u_tilde = vk # Vector 
    u_hat = wk # Vector 
    print("u_tilde ", u_tilde, "\n")
    print("u_hat ", u_hat, "\n")
    theta_tilde = hk/lk # Number 
    r = u_hat - real(theta_tilde)*u_tilde # Vector
    print("r ", r, "\n")

    t = zeros(Float64, rows, 1)

    # Test matrix to see if 
    # conj.(transpose(eig_vect_matrix))*Hk_hat*eig_vect_matrix = A
    eig_vect_matrix = zeros(ComplexF64, rows, cols)
    julia_eigvals = 0
    julia_eigvects = 0

    # Part 2: Inner loop 
    for val = 1:1
        for i = 1:cols  
        # for i = 2:cols # Old version 
            diagonal_A = diag(A)
            A_diagonal_matrix = Diagonal(diagonal_A)
            # print("A_diagonal_matrix ", A_diagonal_matrix, "\n")

            u_tilde_mod = copy(u_tilde)
            # print("u_mod[end:end,1] ", u_mod[end:end,1], "\n")
            u_tilde_mod[end] = 0.0

            u_hat_mod = copy(u_hat)
            # print("u_mod[end:end,1] ", u_mod[end:end,1], "\n")
            u_hat_mod[end] = 0.0

            # Solve for t using bicgstab 

            # 1. Here we using only the diagonal of A and we make the 
            # last element of u_tilde and u_hat equal to 0. 
            t = bicgstab_matrix(((I-(u_tilde_mod*conj.(transpose(u_hat_mod)))/
            (conj.(transpose(u_hat_mod))*u_tilde_mod)[1])*(A_diagonal_matrix-
            real(theta_tilde[1])*I)*(I-(u_tilde_mod*conj.(transpose(u_hat_mod)))/
            (conj.(transpose(u_hat_mod))*u_tilde_mod)[1])),-r)


            # t = bicgstab_matrix(((I-(u_tilde_mod*conj.(transpose(u_hat_mod)))/
            # (conj.(transpose(u_hat_mod))*u_tilde_mod)[1])*(A_diagonal_matrix-
            # real(theta_tilde[1])*I)*(I-(u_tilde_mod*conj.(transpose(u_hat_mod)))/
            # (conj.(transpose(u_hat_mod))*u_tilde_mod)[1])),-r)

            print("t ", t, "\n")
            print("Lk ", Lk[1:i-1,1:i-1], "\n")

            t_tilde = t - Vk[:,1:i-1]*inv(Lk[1:i-1,1:i-1])*
            conj.(transpose(Wk[:,1:i-1]))*t
            print("t_tilde ", t_tilde, "\n")
            vk = t_tilde/norm(t_tilde) # v_{k+1}
            Vk[:,i] = vk # Add the new vk to the Vk matrix -> V_{k+1}

            # New wk that will be added as a new column to Wk
            wk = A*vk # w_{k+1} = A*v_{k+1}
            print("wk ", wk, "\n")
            print("Wk ", Wk, "\n")
            # Expand W_k with this vector to W_{k+1}
            Wk[:,i] = wk # W_{k+1}
            print("new Wk ", Wk, "\n")

            lk = conj.(transpose(wk))*Vk[:,1:i]
            print("lk ", lk, "\n")
            print("Lk ", Lk, "\n")
            # Expand L_k with this vector to W_{k+1}
            Lk[i,1:i] = lk # Row of matrix -> L_{k+1}

            hk = conj.(transpose(wk))*Wk[:,1:i]
            # H_hat_{k+1}
            Hk_hat[i,1:i] = hk # Row of matrix
            Hk_hat[1:i,i] = conj.(transpose(hk)) # Column of matrix
            print("Hk_hat ", Hk_hat, "\n")

            Hk_tilde = inv(Lk[1:i,1:i])*Hk_hat[1:i,1:i] # H_tilde_{k+1}
            
            # julia_eig_solve =  eigsolve(Hk_hat[1:i,1:i]) # Old, this was a mistake
            julia_eig_solve =  eigsolve(Hk_tilde[1:i,1:i])
            julia_eigvals = julia_eig_solve[1]
            julia_eigvects = julia_eig_solve[2]
            print("Julia eigvals ", julia_eigvals, "\n")
            print("Julia eigvectors ", julia_eigvects, "\n")
            theta_tilde = julia_eigvals[end]
            s =  julia_eigvects[end][:] # Minimum eigvector
            print("theta", theta_tilde, "\n")
            print("s", s, "\n")

            # s = eigenvectors
            # theta_tilde = eigvals 
            print("theta (eigenvalue): ", theta_tilde, "\n")
            print("s (eigenvector): ", s, "\n")
            # s = eigenvector/norm(eigenvector) # normalized_eigenvector
            # print("s ", s, "\n")
            
            # Compute the harmonic Ritz vector 
            u_tilde = (Vk[:,1:i]*s)/norm(Vk[:,1:i]*s)
            u_hat = A*u_tilde 
            # Should be equal to (Wk*s)/norm(Vk*s)
            
            # Compute the residual 
            r = u_hat - theta_tilde[1]*u_tilde # Residual vector 
            print("r ", r, "\n")
            print("norm of residual ", norm(r), "\n")
            print("real((conj.(transpose(r))*r)[1]) ", 
            real((conj.(transpose(r))*r)[1]), "\n")

            # if norm(r) <= tol
            if real((conj.(transpose(r))*r)[1]) < tol
                print("Exited the loop using break")
                break
            end 
            
        end

        print("Vk*conj.(transpose(Vk)) ", conj.(transpose(Vk))*Vk, "\n")
        
        # Test 
        # print("julia_eigvects[:][:] ", julia_eigvects[:][:], "\n")
        # # eig_vect_matrix[:,:] = julia_eigvects[:][:]

        # eig_vect_matrix[:,1] = julia_eigvects[1][:]
        # eig_vect_matrix[:,2] = julia_eigvects[2][:]
        # eig_vect_matrix[:,3] = julia_eigvects[3][:]
        # print("eig_vect_matrix ", eig_vect_matrix, "\n")
        # # A_test = eig_vect_matrix*Hk_tilde*eig_vect_matrix
        # A_test = eig_vect_matrix*Hk_tilde*conj.(transpose(eig_vect_matrix))
        # # A_test = conj.(transpose(eig_vect_matrix))*Hk_tilde*eig_vect_matrix
        # print("A_test ", A_test, "\n")
        # print("A ", A, "\n")
    end
    return real(theta_tilde)
end 

# We want to get 0.885092 as our minimum eigenvalue and (5.58774,3.11491,1)
# as our eigenvector associate to the minimum eigenvalue

A = Array{ComplexF64}(undef, 2, 2)
A[1,1] = 3+0im
A[1,2] = 1-1im
A[2,1] = 1+5im
A[2,2] = -2+3im
# A[1,1] = 0.6631086137342226 + 0.4748299965984625im 
# A[1,2] = 0.9249038573578873 + 0.33631052040436304im
# A[2,1] = 0.7833453482091092 + 0.792496415913588im 
# A[2,2] = 0.12232749282943689 + 0.2285817823475802im




# rand!(A)

# A = Array{ComplexF64}(undef, 3, 3)
# A[1,1] = 2+2im
# A[1,2] = -1+3im
# A[1,3] = 0+0im
# A[2,1] = -1+3im
# A[2,2] = 2-2im
# A[2,3] = -1-1im
# A[3,1] = 0-0im
# A[3,2] = -1+2im
# A[3,3] = 2+3im
# A[1,1] = 2.0
# A[1,2] = -2.0
# A[1,3] = 0.0
# A[2,1] = -1.0
# A[2,2] = 3.0
# A[2,3] = -1.0 
# A[3,1] = 0.0
# A[3,2] = -1.0
# A[3,3] = 4.0
A = (A.+conj.(transpose(A)))./2

# test
# A = [2.0 + im*0.0  -2.0 + im*0.0  0.0 + im*0.0;
# 	-1.0 + im*0.0  3.0 + im*0.0  -1.0 + im*0.0;
# 	0.0 + im*0.0  -1.0 + im*0.0  4.0 + im*0.0]

# A[:,:] = (A .+ conj(transpose(A))) ./ 2


# A = Array{ComplexF64}(undef, 2,2)
# rand!(A)
# A[:,:] = (A .+ conj(transpose(A))) ./ 2
# print("det(A) ", det(A),"\n")



print("A ", A, "\n")

# Generate random large matrix 
# A = Array{Float64}(undef, 100,100)
# A[:,:] = (A .+ conj(transpose(A))) ./ 2
# rand!(A)

print("Davidson code ",davidson_it(A), "\n")
print("Julia direct solve ",eigen(A).values, "\n")
end 





