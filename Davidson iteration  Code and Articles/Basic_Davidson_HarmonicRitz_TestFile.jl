module Davidson_HarmonizRitz_TestFile
using LinearAlgebra, Random, Arpack, KrylovKit, bicgstab, cg

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

function modified_gram_schmidt!(Vk,n,t,Wk_tilde)
    num_vectors = size(Vk)[2] 
    tol = 1e-5
    nrm = norm(Vk[:,n])
    v_vect = t/norm(t) # Normalize t 
    print("n ", n, "\n")
    print("v_vect ", v_vect, "\n")
    # print("norm(t) ", norm(t), "\n")

    # if nrm < tol
    #     # Vk[:,n] = Vk[:,n]+rand(size(Vk)[2],1)
    #     Vk[:,n] = Vk[:,n]+rand(size(Vk)[2],1)
    #     modified_gram_schmidt!(Vk,n,t,Wk_tilde)
    # end 

    for index = 1:n-1
        v_vect = v_vect - Vk[:,index]*conj.(transpose(Wk_tilde[:,index]))*v_vect
        v_vect = v_vect/norm(v_vect)
        print("norm(v_vect) ", norm(v_vect), "\n")
        # Vk[:,n] = Vk[:,n] - conj.(transpose((Vk[:,1:n-1])))*Vk[:,n]
    end 
    # After this loop we get t_tilde = v_vect
    # Instead of using a variable to store this, we directly put it in Vk[:,n]
    Vk[:,n] = v_vect
    # nrm = norm(Vk[:,n])
    # if nrm < tol
    #     Vk[:,n] = Vk[:,n]+rand(size(Vk)[2],1)
    #     modified_gram_schmidt!(Vk,n,t,wk_tilde)
    # end 
    # Vk[:,n] = Vk[:,n]/nrm

    # The prints below are checks that everything works as it should 
    print("norm(Vk[:,n]) ", norm(Vk[:,n]), "\n")

    print("Vk[:,1]: ", Vk[:,1], "\n")
    print("Vk[:,2]: ", Vk[:,2], "\n")
    print("Vk[:,3]: ", Vk[:,3], "\n")

    print("conj.(transpose(Vk)).*Vk: ", conj.(transpose(Vk)).*Vk, "\n")
    print("conj.(transpose(Vk))*Vk: ", conj.(transpose(Vk))*Vk, "\n")
    print("Vk dot Vk: ", dot(Vk,Vk), "\n")
    print("conj(transpose(Vk))*Vk: ", conj(transpose(Vk))*Vk, "\n")

    print("1 dot 2: ", dot(Vk[:,1],Vk[:,2]), "\n")
    print("1 dot 3: ", dot(Vk[:,1],Vk[:,3]), "\n")
    print("2 dot 3: ", dot(Vk[:,2],Vk[:,3]), "\n")

    print("Vk[:,1]", norm(Vk[:,1]), "\n")
    print("Vk[:,2]", norm(Vk[:,2]), "\n")
    print("Vk[:,3]", norm(Vk[:,3]), "\n")

end


function davidson_it(A)
    """
    This function is the Davidson iteration algorithm. It is a hopefully better
    alternative to the Power iteration method since it uses past information to 
    choose search directions and at each new iteration. We use harmonic Rizt 
    vectors here, which allows to directly find the minimum eigenvalue. 
    """
    # m = 20 # Amount of iterations of the inner loop

    # Part 1. Setup 
    tol = 1e-10 # Tolerance for which the program will converge 

    rows = size(A)[1]
    cols = size(A)[2]

    v = zeros(ComplexF64, rows, 1)
    rand!(v)
    # print("v ", v, "\n")

    vk = v/norm(v) # Vector 
    wk = A*v # Vector 
    lk = (conj.(transpose(wk))*vk)[1] # Number
    hk = (conj.(transpose(wk))*wk)[1] # Number
    # print("hk ", hk, "\n")

    # Vk = Array{Float64}(undef, rows, cols)
    Vk = zeros(ComplexF64, rows, cols)
    # The first index is to select a row
    # The second index is to select a column
    Vk[:,1] = vk
    # print("Vk ", Vk, "\n")

    # Wk = Array{Float64}(undef, rows, cols)
    Wk = zeros(ComplexF64, rows, cols)
    Wk[:,1] = wk
    # print("Wk ", Wk, "\n")
    
    Lk = zeros(ComplexF64, rows, cols)
    # Lk = Array{Float64}(undef, rows, cols)
    Lk[1,1] = lk
    # print("Lk ", Lk, "\n")

    Wk_tilde = zeros(ComplexF64, rows, cols)
    Wk_tilde[:,1] = wk*inv(Lk[1,1])
    # Wk_tilde = Wk * inv(Lk)
    # print("Wk_tilde ", Wk_tilde, "\n")

    Hk_hat = zeros(ComplexF64, rows, cols)
    # Hk_hat = Array{Float64}(undef, rows, cols)
    Hk_hat[1,1] = hk
    # print("Hk_hat ", Hk_hat, "\n")

    Hk_tilde = zeros(ComplexF64, rows, cols)

    u_tilde = vk # Vector 
    u_hat = wk # Vector 
    # print("u_tilde ", u_tilde, "\n")
    # print("u_hat ", u_hat, "\n")
    theta_tilde = hk/lk # Number 
    r = u_hat - real(theta_tilde)*u_tilde # Vector
    # print("r ", r, "\n")

    t = zeros(ComplexF64, rows, 1)

    # Test matrix to see if 
    # conj.(transpose(eig_vect_matrix))*Hk_hat*eig_vect_matrix = A
    eig_vect_matrix = zeros(ComplexF64, rows, cols)
    julia_eigvals = 0
    julia_eigvects = 0

    # Part 2: Inner loop 
    for i = 2:cols # Iterate through all of the columns 
        print("i ", i, "\n")

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
        (conj.(transpose(u_hat_mod))*u_tilde_mod)[1])*(A-
        real(theta_tilde[1])*I)*(I-(u_tilde_mod*conj.(transpose(u_hat_mod)))/
        (conj.(transpose(u_hat_mod))*u_tilde_mod)[1])),-r)

        """
        Biorthogonolize t against Vk and Wk method .
        This method becomes very unstable for large matrices.
        We therefore need to use another method, which can be done by using 
        Wk_tilde = Wk*(L_k^*)^{-1}. This method allows us to use modified 
        Gram-Schmidt as an orthogonalization method. 
        """
        # Here's the other method to find t that should be more stable for 
        # larger systems.
        # Instead of explicity having a variable to t_tilde and then normalizing
        # it, I add the already normalized vector to Vk.
        modified_gram_schmidt!(Vk,i,t,Wk_tilde)

        # Here's the basic method 
        # t_tilde = t - Vk[:,1:i-1]*inv(Lk[1:i-1,1:i-1])*
        # conj.(transpose(Wk[:,1:i-1]))*t
        # print("t_tilde ", t_tilde, "\n")
        # vk = t_tilde/norm(t_tilde) # v_{k+1}
        # Vk[:,i] = vk # Add the new vk to the Vk matrix -> V_{k+1}

        # New wk that will be added as a new column to Wk
        wk_tilde = A*Vk[:,i] # w_{k+1} = A*v_{k+1}
        # wk_tilde = A*vk # w_{k+1} = A*v_{k+1}
        # print("wk ", wk, "\n")
        # print("Wk ", Wk, "\n")
        # Expand W_k with this vector to W_{k+1}
        Wk_tilde[:,i] = wk_tilde # W_{k+1}
        # print("new Wk ", Wk, "\n")

        lk = conj.(transpose(wk_tilde))*Vk[:,1:i]
        # print("lk ", lk, "\n")
        # print("Lk ", Lk, "\n")
        # Expand L_k with this vector to W_{k+1}
        Lk[i,1:i] = lk # Row of matrix -> L_{k+1}

        hk = conj.(transpose(wk_tilde))*Wk_tilde[:,1:i]
        # H_hat_{k+1}
        Hk_hat[i,1:i] = hk # Row of matrix
        Hk_hat[1:i,i] = conj.(transpose(hk)) # Column of matrix
        # print("Hk_hat ", Hk_hat, "\n")

        Hk_tilde = inv(Lk[1:i,1:i])*Hk_hat[1:i,1:i] # H_tilde_{k+1}

        print("size(Hk_tilde) ", size(Hk_tilde), "\n")
        

        # julia_eig_solve =  eigsolve(Hk_hat[1:i,1:i]) # Old, this was a mistake
        julia_eig_solve =  eigsolve(Hk_tilde[1:i,1:i])
        julia_eigvals = julia_eig_solve[1]
        julia_eigvects = julia_eig_solve[2]

        print("julia_eigvals ", julia_eigvals, "\n")
        min_eigval = 1000
        position = 1
        for i in eachindex(julia_eigvals)
            if 0 < real(julia_eigvals[i]) < min_eigval
                theta_tilde = real(julia_eigvals[i])
                position = i 
                min_eigval = real(theta_tilde)
                # print("position ", position, "\n")
                # print("Updated position ", i, " times \n")
            end 
            # print("i ", i, "\n")
            # print("min_eigval ", min_eigval, "\n")
        end 
        # s =  julia_eigvects[position][:] # Minimum eigvector
        # print("s1 ", s, "\n")
        s =  julia_eigvects[:][position] # Minimum eigvector


        print("Julia eigvals ", julia_eigvals, "\n")
        print("Julia eigvectors ", julia_eigvects, "\n")
        # theta_tilde = julia_eigvals[end]
        # s =  julia_eigvects[end][:] # Minimum eigvector
        print("theta_tilde ", theta_tilde, "\n")
        print("s ", s, "\n")
        
        # Compute the harmonic Ritz vector 
        u_tilde = (Vk[:,1:i]*s)/norm(Vk[:,1:i]*s)
        u_hat = A*u_tilde 
        # Should be equal to (Wk*s)/norm(Vk*s)
        
        # Compute the residual 
        r = u_hat - theta_tilde[1]*u_tilde # Residual vector 
        # print("r ", r, "\n")
        # print("norm of residual ", norm(r), "\n")
        # print("real((conj.(transpose(r))*r)[1]) ", 
        # real((conj.(transpose(r))*r)[1]), "\n")

        # if norm(r) <= tol
        if real((conj.(transpose(r))*r)[1]) < tol
            print("Exited the loop using break \n")
            break
        end 
        
    end
    print("conj(transpose(Vk))*Vk ", conj(transpose(Vk))*Vk, "\n")
    return real(theta_tilde)
end 

# We want to get 0.885092 as our minimum eigenvalue and (5.58774,3.11491,1)
# as our eigenvector associate to the minimum eigenvalue

# A = Array{Float64}(undef,50,50)
A = Array{ComplexF64}(undef,3,3)
A[1,1] = 2
A[1,2] = -2
A[1,3] = 0
A[2,1] = -1
A[2,2] = 3
A[2,3] = -1 
A[3,1] = 0
A[3,2] = -1
A[3,3] = 4
# rand!(A)
# A = (A+conj.(transpose(A)))/2
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
# print("A ", A, "\n")
# print(A[1,:])
# print("size(A) ", size(A)[2], "\n")

# print(eigen(A).values)

print("Davidson minimum positive eigenvalue ",davidson_it(A), "\n")
print("Julia direct solve ",eigen(A).values, "\n")
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