module Davidson_HarmonizRitz_TestFile
using LinearAlgebra, Random, Arpack, KrylovKit # , bicgstab

# function modified_gram_schmidt(A, t, index) # i
#     # orthogonalises the columns of the input matrix
#     matrix = Array{Float64}(undef, size(A)[1],index+1) # 1rst old thing index
# +size(t)[2] , other old thing size(A)[1]
#     # i is the amount of cols (which are vectors) of A that we are 
# considering here 
#     matrix[:,1:index] = A #  matrix[:,1:size(A)[2]] = A
#     matrix[:,index+1:end] = t

#     num_vectors = size(matrix)[2]
#     orth_matrix = copy(matrix)
#     for vec_idx = 1:num_vectors
#         orth_matrix[:, vec_idx] = orth_matrix[:, vec_idx]/norm(orth_matrix[:, vec_idx])
#         for span_base_idx = (vec_idx+1):num_vectors
#             # perform block step
#             orth_matrix[:, span_base_idx] -= dot(orth_matrix[:, span_base_idx], orth_matrix[:, vec_idx])*orth_matrix[:, vec_idx]
#         end
#     end
#     return orth_matrix[:,end:end]
# end

# function gram!(V) # i
#     print("V MGS", V, "\n")
#     nrm = norm(V[:,end])
#     V[:,end] = V[:,end]/nrm
#     print("V[:,1:end-1] ", V[:,1:end-1], "\n")
#     print("V[:,end] ", V[:,end], "\n")
#     # proj_coeff = dot(V[:,1:end-1],V[:,end])
#     proj_coeff = conj!(tranpspose([:,1:end-1])).*V[:,end]
#     V[:,end] = V[:,end] - proj_coeff*V[:,1:end-1]
#     nrm = norm(V[:,end])
#     V[:,end] = V[:,end]/nrm
# end


function davidson_it(A)
    # m = 20 # Amount of iterations of the inner loop

    # Part 1. Setup 
    tol = 1e-15 # Tolerance for which the program will converge 

    rows = size(A)[1]
    cols = size(A)[2]

    v = zeros(Float64, rows, 1)
    rand!(v)
    print("v ", v, "\n")

    vk = v/norm(v) # Vector 
    wk = A*v # Vector 
    lk = (conj.(transpose(wk))*vk)[1] # Number
    hk = (conj.(transpose(wk))*wk)[1] # Number
    print("hk ", hk, "\n")

    # Vk = Array{Float64}(undef, rows, cols)
    Vk = zeros(Float64, rows, cols)
    # The first index is to select a row
    # The second index is to select a column
    Vk[:,1] = vk
    print("Vk ", Vk, "\n")

    # Wk = Array{Float64}(undef, rows, cols)
    Wk = zeros(Float64, rows, cols)
    Wk[:,1] = wk
    print("Wk ", Wk, "\n")
    
    Lk = zeros(Float64, rows, cols)
    # Hk = Array{Float64}(undef, rows, cols)
    Lk[1,1] = lk
    print("Lk ", Lk, "\n")

    Hk = zeros(Float64, rows, cols)
    # Hk = Array{Float64}(undef, rows, cols)
    Hk[1,1] = hk
    print("Hk ", Hk, "\n")

    u_tilde = vk # Vector 
    u_hat = wk # Vector 
    theta_tilde = hk/lk # Number 
    r = u_hat - real(theta_tilde)*u_tilde # Vector
    print("r ", r, "\n")

    # t = Array{Float64}(undef, rows, 1)
    t = zeros(Float64, rows, 1)

    # u_hat = Array{Float64}(undef, rows, 1)
    # u_hat = zeros(Float64, rows, 1)

    # Part 2: Inner loop 
    for val = 1:10
        for i = 2:cols # Old version 
            diagonal_A = diag(A)
            A_diagonal_matrix = Diagonal(diagonal_A)
            # print("A_diagonal_matrix ", A_diagonal_matrix, "\n")

            u_tilde_mod = copy(u_tilde)
            # print("u_mod[end:end,1] ", u_mod[end:end,1], "\n")
            u_tilde_mod[end] = 0.0

            u_hat_mod = copy(u_hat)
            # print("u_mod[end:end,1] ", u_mod[end:end,1], "\n")
            u_hat_mod[end] = 0.0
            
            # print("u ", u, "\n")
            # print("u_mod ", u_mod, "\n")
            
            # print("I-u_mod*conj.(transpose(u_mod)) ", I-u_mod*conj.(transpose(u_mod)),"\n")
            # print("A_diagonal_matrix-theta[1]*I ", A_diagonal_matrix-theta[1]*I,"\n")
            # print("I-u_mod*conj.(transpose(u_mod)) ", I-u_mod*conj.(transpose(u_mod)),"\n")

            # print("conj.(transpose(u_hat_mod)) ", 
            # conj.(transpose(u_hat_mod)), "\n")
            # print("u_tilde_mod ", u_tilde_mod, "\n")

            # print("conj.(transpose(u_hat_mod))*u_tilde_mod ", 
            # conj.(transpose(u_hat_mod))*u_tilde_mod, "\n")

            
            t = inv(((I-(u_tilde_mod*conj.(transpose(u_hat_mod)))/
            (conj.(transpose(u_hat_mod))*u_tilde_mod)[1])*(A_diagonal_matrix-
            real(theta_tilde[1])*I)*(I-(u_tilde_mod*conj.(transpose(u_hat_mod)))/
            (conj.(transpose(u_hat_mod))*u_tilde_mod)[1])))

            print("t ", t, "\n")
            print("Lk ", Lk[1:i-1,1:i-1], "\n")

            t_tilde = t - Vk*inv(Lk[1:i-1,1:i-1])*conj.(transpose(Wk))*t
            vk = t_tilde/norm(t_tilde)
            Vk[:,i] = vk # Add the new vk to the Vk matrix

            # New wk that will be added as a new column to Wk
            wk = A*vk 
            print("wk ", wk, "\n")
            print("Wk ", Wk, "\n")
            # Expand W_k with this vector to W_{k+1}
            Wk[:,i] = wk 
            print("new Wk ", Wk, "\n")

            lk = conj.(tranpose(wk))*Vk
            # Expand L_k with this vector to W_{k+1}
            Lk[i,1:i] = lk # Row of matrix 

            hk = conj.(transpose(wk))*Wk
            Hk[i,1:i] = hk # Row of matrix
            Hk[1:i,i] = conj.(transpose(hk)) 

            # print("Vk ", Vk, "\n")
            # print("Vk[:,1:i] ", Vk[:,1:i], "\n")
            # print("wk ", wk, "\n")
            # # V*_{k+1} wk = V*_{k+1} A V*_{k+1} = V*_{k+1} A V*_{k+1}
            # # print("(conj.(transpose(Vk))*wk)[1] ", (conj.(transpose(Vk))*wk)[1], "\n")

            # # Hk[i,i] = (conj.(transpose(Vk))*wk)[1]
            # print("OG Hk ", Hk, "\n")
            # # hk is now a vector
            # hk = (conj.(transpose(Vk[:,1:i]))*wk)
            # Hk[1:i,i] = hk # Column of matrix 
            # # Hk[1:i,i] = (conj.(transpose(Vk[:,1:i]))*wk)
            # print("new Hk v1 ", Hk, "\n")

            # If we assume that A is hermitan, we don't have index+1to calculate vk*Wk
            # for the last row of Hk. We can just take the complex conjugate of 
            # Vk*wk, which we just caculated.
            # Hk[i,1:i] = conj(transpose(hk)) # Row of matrix
            # print("new Hk v2 ", Hk, "\n")
            
            print("Julia eigvals ", eigsolve(Hk[1:i,1:i]), "\n")

            # Compute the largest eigenpar (theta,s) of H_{k+1} 
            # with the norm(s) = 1
            eigvals, eigenvectors = eigs(Hk[1:i,1:i], which=:SM) 
            # SM => smallest magnitude 
            print("eigvals ", eigvals, "\n")
            print("eigenvectors ", eigenvectors, "\n")

            # We only want the eigenvalue that is positive
            position = 0

            min_eigvals = 0.0 
            for i in eachindex(eigvals) # We need the eigenvalues to be positive, right (like before)?
                if eigvals[i] > 0 
                    if eigvals[i] < min_eigvals
                        min_eigvals = eigvals[i]
                        theta_tilde = eigvals[i]
                        position = i 
                    end 
                    print("position ", position, "\n")
                end 
            end 
            s = eigenvectors[:,position]
            print("theta (eigenvalue): ", theta_tilde, "\n")
            print("s (eigenvector): ", s, "\n")
            # s = eigenvector/norm(eigenvector) # normalized_eigenvector
            # print("s ", s, "\n")
            
            # Compute the harmonic Ritz vector 
            u_tilde = (Vk*s)/norm(Vk*s)
            u_hat = A*u_tilde 
            # Should be equal to (Wk*s)/norm(Vk*s)
            
            # Compute the residual 
            r = u_hat - theta_tilde[1]*u_tilde # Residual vector 
            print("r ", r, "\n")
            
            # print("Vk ", Vk, "\n")
            # u = Vk[:,1:i]*s # Compute the Ritz vector u  
            # print("u ", u, "\n")
            # u_hat = A*u # Should also have: A*u = W_k*s

            # # u_hat test to see if A*u = W_k*s 
            # u_hat_test = Wk[:,1:i]*s
            # print("u_hat_test ", u_hat_test, "\n")

            # print("u_hat ", u_hat, "\n")
            # print("theta[1] ", real(theta[1]), "\n")
            # print("theta[1]*u ", theta[1]*u, "\n")
            # r = u_hat - theta[1]*u # Residual vector 
            # print("r ", r, "\n")
            
        end
        print("norm of residual ", norm(r), "\n")
        if norm(r) <= tol
            print("Exited the loop using break")
            break 
        end 
        # Vk = zeros(Float64, rows, cols)
        # Vk[1:end,1] = u

        # Wk = zeros(Float64, rows, cols)
        # Wk[1:end,1] = u_hat
        
        # Hk = zeros(Float64, rows, cols)
        # Hk[1,1] = real(theta[1])
        
    end
    return real(theta)
end 

A = Array{Float64}(undef, 3, 3)
A[1,1] = 2
A[1,2] = -2
A[1,3] = 0
A[2,1] = -1
A[2,2] = 3
A[2,3] = -1 
A[3,1] = 0
A[3,2] = -1
A[3,3] = 4
# A = zeros(Int8, 3, 3)
# A[1,1] = 2
# A[1,2] = -1
# A[1,3] = 0
# A[2,1] = -1
# A[2,2] = 2
# A[2,3] = -1 
# A[3,1] = 0
# A[3,2] = -1
# A[3,3] = 2
print("A ", A, "\n")
# print(A[1,:])
# print("size(A) ", size(A)[2], "\n")

# print(eigen(A).values)

print(davidson_it(A), "\n")
end 






    # # t = inv(((I-u_mod*conj.(transpose(u_mod)))*(A_diagonal_matrix-
    # # real(theta[1])*I)*(I-u_mod*conj.(transpose(u_mod)))))*(-r)
    
    # # print("size(r) ", size(r), "\n")
    # # print("size(t) ", size(t), "\n")


    # # Before, the code took Vk without the n+1 column, which is 
    # # what we want to replace with a version of t that is orthogonal 
    # # to all the columns of Vk that is calculated with modified 
    # # gramd schmidth) and used the t calculated above in the MGS 
    # # function. This was inefficient because we would create a new 
    # # matrix that would combine Vk and t into one matrix each time 
    # # the function was called. Instead, here we will directly replace 
    # # the n+1 column by t and we will modify the memory associated to Vk
    # # in the new modified gram schmidth function that was suggested by 
    # # Sean. This is way more efficient since we only defined Vk once 
    # # and we just modify the Vk's memory. 
    # print("Vk ", Vk, "\n")
    # print("t ", t, "\n")
    # Vk[:,i] = t
    # print("V Loop", Vk, "\n")
    # # print("Index test for MGS ", Vk[:,1:end], "\n")
    # gram!(view(Vk,:,1:i))

    # # index = i-1
    # # print("Vk ", Vk, "\n")
    # # print("Vk[:,index] ", Vk[:,1:index], "\n")
    # # # New vk vector that will be added as a new column to Vk
    # # vk = modified_gram_schmidt(Vk[:,1:index], t, index) # modified_gram_schmidt(Vk, t, i)
    # # print("vk ", vk, "\n")
    # # # Expand V_k with this vector to V_{k+1}
    # # Vk[:,i] = vk
    # # print("new Vk ", Vk, "\n")
    # # # V_{k+1} = vk -> False! See above.