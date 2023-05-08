module davidson_TestFile_New
using LinearAlgebra, Random, Arpack, bicgstab
using KrylovKit

# function modified_gram_schmidt(A, t, index) # i
#     # orthogonalises the columns of the input matrix
#     matrix = Array{Float64}(undef, size(A)[1],index+1) # 1rst old thing index+size(t)[2] , other old thing size(A)[1]
#     # i is the amount of cols (which are vectors) of A that we are considering here 
#     matrix[:,1:index] = A #  matrix[:,1:size(A)[2]] = A
#     matrix[:,index+1:end] = t

#     num_vectors = size(matrix)[2]
#     orth_matrix = copy(matrix)
#     for vec_idx = 1:num_vectors
#         orth_matrix[:, vec_idx] = orth_matrix[:, vec_idx]/norm(orth_matrix[:, vec_idx])
#         for span_base_idx = (vec_idx+1):num_vectors
#             # perform block step
#             orth_matrix[:, span_base_idx] -= dot(orth_matrix[:, span_base_idx], 
#             orth_matrix[:, vec_idx])*orth_matrix[:, vec_idx]
#         end
#     end
#     return orth_matrix[:,end:end]
# end

function modified_gram_schmidt!(Vk,n)
    num_vectors = size(Vk)[2] 
    tol = 1e-6
    nrm = norm(Vk[:,n])
    if nrm < tol
        Vk[:,n] = Vk[:,n]+rand(size(matrix)[2],1)
        modified_gram_schmidt!(Vk,n)
    end 
    # for index = 1:num_vectors
    #     Vk[:,index] = Vk[:,n] - conj.(transpose((Vk[:,1:n-1])))*Vk[:,n]
    # end 

    Vk[:,n] = Vk[:,n] - (Vk[:,1:n-1])*((conj.(transpose((Vk[:,1:n-1])))*Vk[:,n]))

    # for vec_idx = 1:num_vectors
    #     # Vk[:, vec_idx] = Vk[:, vec_idx]/norm(Vk[:, vec_idx])
    #     for span_base_idx = (vec_idx+1):num_vectors
    #         # perform block step
    #         Vk[:, span_base_idx] -= dot(Vk[:, span_base_idx], 
    #         Vk[:, vec_idx])*Vk[:, vec_idx]
    #     end
    # end

    # Vk[:,n] = Vk[:,n] - conj.(transpose((Vk[:,1:n-1])))*Vk[:,n]
    nrm = norm(Vk[:,n])
    if nrm < tol
        Vk[:,n] = Vk[:,n]+rand(size(Vk)[2],1)
        modified_gram_schmidt!(Vk,n)
    end 
    Vk[:,n] = Vk[:,n]/nrm

    print("Vk[:,1]: ", Vk[:,1], "\n")
    print("Vk[:,2]: ", Vk[:,2], "\n")
    print("Vk[:,3]: ", Vk[:,3], "\n")

    print("conj.(transpose(Vk)).*Vk: ", conj.(transpose(Vk)).*Vk, "\n")
    print("conj.(transpose(Vk))*Vk: ", conj.(transpose(Vk))*Vk, "\n")
    print("Vk dot Vk: ", dot(Vk,Vk), "\n")

    print("1 dot 2: ", dot(Vk[:,1],Vk[:,2]), "\n")
    print("1 dot 3: ", dot(Vk[:,1],Vk[:,3]), "\n")
    print("2 dot 3: ", dot(Vk[:,2],Vk[:,3]), "\n")

    print("Vk[:,1]", norm(Vk[:,1]), "\n")
    print("Vk[:,2]", norm(Vk[:,2]), "\n")
    print("Vk[:,3]", norm(Vk[:,3]), "\n")

end

function gramSchmidt!(basis::Array{T}, n::Integer) where T <: Number
    tol = 1e-6
	# dimension of vector space
	dim = size(basis)[1]
	# orthogonality check
	prjNrm = 1.0;
	# check that basis does not exceed dimension
	if n > dim
		error("Requested basis size exceeds a postdoctoral research associate 
		at Princeton dimension of vector space.")
	end
	# norm calculation
	nrm = BLAS.nrm2(dim, view(basis,:,n), 1)
	# renormalize new vector
	basis[:,n] = basis[:,n] ./ nrm
	nrm = BLAS.nrm2(dim, view(basis,:,n), 1)
	# guarded orthogonalization
	while prjNrm > (tol * 100) && abs(nrm) > tol
		# remove projection into existing basis
 		BLAS.gemv!('N', -1.0 + im*0.0, view(basis, :, 1:(n-1)), 
 			BLAS.gemv('C', view(basis, :, 1:(n-1)), view(basis, :, n)), 
 			1.0 + im*0.0, view(basis, :, n)) 
 		# recalculate the norm
 		nrm = BLAS.nrm2(dim, view(basis,:,n), 1) 
 		# calculate projection norm
 		prjNrm = BLAS.nrm2(n-1, BLAS.gemv('C', 
 			view(basis, :, 1:(n-1)), view(basis, :, n)), 1) 
 	end
	# check that remaining vector is sufficiently large
	if abs(nrm) < tol
		# switch to random basis vector
		rand!(view(basis, :, n))
		gramSchmidt!(basis, n)		
	else
		# renormalize orthogonalized vector
		basis[:,n] = basis[:,n] ./ nrm
	end 
end


function davidson_it(A)
    # m = 20 # Amount of iterations of the inner loop

    tol = 1e-3 # Tolerance for which the program will converge 

    rows = size(A)[1]
    cols = size(A)[2]

    # Set starting vector
    v = zeros(ComplexF64, rows, 1)
    v = [0.9026710530499227 + 0.8801240709989707im;
    0.8797904471904741 + 0.13830203407554031im; 
    0.3804996099871757 + 0.19827553581504442im]
    # rand!(v)
    print("v ", v, "\n")

    # Normalize starting vector
    vk = v/norm(v) # Vector 
    wk = A*vk # Vector 
    print("wk ", wk, "\n")
    hk = BLAS.dotc(cols, vk, 1, wk, 1) 
    # hk = (conj.(transpose(vk))*wk)[1] # Number
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
    
    # Hessenberg matrix 
    Hk = zeros(ComplexF64, rows, cols)
    # Hk = Array{Float64}(undef, rows, cols)
    Hk[1,1] = hk
    print("Hk ", Hk, "\n")

    # Ritz vector 
    u = vk # Vector
    # Eigenvalue  
    theta = hk # Number 
    print("theta ", theta, "\n")
    # Negative residual vector 
    r = real(theta).*u .- wk # Vector
    # Residual vector 
    # r = wk - real(theta).*u # Vector
    print("r ", r, "\n")

    s = zeros(ComplexF64, rows, 1)

    # t = Array{Float64}(undef, rows, 1)
    t = zeros(ComplexF64, rows, 1)

    # u_hat = Array{Float64}(undef, rows, 1)
    u_hat = zeros(Float64, rows, 1)

    # Test matrix to see if 
    # conj.(transpose(eig_vect_matrix))*Hk*eig_vect_matrix = A
    eig_vect_matrix = zeros(Float64, rows, cols)
    julia_eigvals = 0
    julia_eigvects = 0

    # for val = 1:1
    for i = 2:cols 
        # diagonal_A = diag(A)
        # A_diagonal_matrix = Diagonal(diagonal_A)
        # print("A_diagonal_matrix ", A_diagonal_matrix, "\n")

        u_mod = copy(u)
        # print("u_mod[end:end,1] ", u_mod[end:end,1], "\n")
        u_mod[end] = 0.0
        # print("u ", u, "\n")
        
        # print("u_mod ", u_mod, "\n")
        
        # print("I-u_mod*conj.(transpose(u_mod)) ", I-u_mod*conj.(transpose(u_mod)),"\n")
        # print("A_diagonal_matrix-theta[1]*I ", A_diagonal_matrix-theta[1]*I,"\n")
        # print("I-u_mod*conj.(transpose(u_mod)) ", I-u_mod*conj.(transpose(u_mod)),"\n")

        # Solve using bicgstab
        # This is t in the algorithm 
        # bicgstab_matrix(A, theta, u, b)
        # t = bicgstab_matrix(((I-u_mod*conj.(transpose(u_mod)))*
        # (A-real(theta[1])*I)*
        # (I-u_mod*conj.(transpose(u_mod)))),theta,u,r) # My code 

        print("A check ", A, "\n")
        print("theta check ", theta, "\n")
        print("u check ", u, "\n")
        print("r check ", r, "\n")

        t = bicgstab_matrix_ritz(A,theta,u,r) # Sean's code 
        # t = bicgstab_matrix(((I-u_mod*conj.(transpose(u_mod)))*
        # (A_diagonal_matrix-real(theta[1])*I)*
        # (I-u_mod*conj.(transpose(u_mod)))),-r)

        print("tsolve ", t, "\n")

        # Add the t vector as the latest column of Vk 
        Vk[:,i] = t/norm(t)

        # Solve using inverse 
        # t = inv(((I-u_mod*conj.(transpose(u_mod)))*(A_diagonal_matrix-real(theta[1])*I)*(I-u_mod*conj.(transpose(u_mod)))))*(-r)
        
        # print("size(r) ", size(r), "\n")
        # print("size(t) ", size(t), "\n")
    
        # index = i-1
        # modified_gram_schmidt!(Vk, i) # My code
        gramSchmidt!(Vk, i) # Sean's code 

        # index = i-1
        # print("Vk ", Vk, "\n")
        # print("Vk[:,index] ", Vk[:,1:index], "\n")
        # New vk vector that will be added as a new column to Vk
        # vk = modified_gram_schmidt(Vk[:,1:index], t, index) # modified_gram_schmidt(Vk, t, i)
        # print("vk ", vk, "\n")
        # # Expand V_k with this vector to V_{k+1}
        # Vk[:,i] = vk
        # print("new Vk ", Vk, "\n")
        # V_{k+1} = vk -> False! See above.

        # New wk that will be added as a new column to Wk
        wk = A*Vk[:,i]
        # print("wk ", wk, "\n")
        # print("Wk ", Wk, "\n")
        # Expand W_k with this vector to W_{k+1}
        Wk[:,i] = wk 
        print("new Wk ", Wk, "\n")

        # print("Vk ", Vk, "\n")
        # print("Vk[:,1:i] ", Vk[:,1:i], "\n")
        # print("wk ", wk, "\n")
        # V*_{k+1} wk = V*_{k+1} A V*_{k+1} = V*_{k+1} A V*_{k+1}
        # print("(conj.(transpose(Vk))*wk)[1] ", (conj.(transpose(Vk))*wk)[1], "\n")

        # Hk[i,i] = (conj.(transpose(Vk))*wk)[1]
        # print("OG Hk ", Hk, "\n")
        # hk is now a vector
        hk = (conj.(transpose(Vk[:,1:i]))*wk)
        Hk[1:i,i] = hk # Column vector 
        # Hk[1:i,i] = (conj.(transpose(Vk[:,1:i]))*wk)
        # print("new Hk v1 ", Hk, "\n")

        # If we assume that A is hermitan, we don't have index+1to calculate vk*Wk
        # for the last row of Hk. We can just take the complex conjugate of 
        # Vk*wk, which we just caculated.
        Hk[i,1:i] = conj(transpose(hk)) # Row vector 
        # Hk[i,1:i] = conj(transpose((conj.(transpose(Vk[:,1:i]))*wk)))
        # print("new Hk v2 ", Hk, "\n")

        # We comment this part for debugging purposes
        # if A != conj((transpose(A)))
        #     print("(conj.(transpose(Vk))*wk)[1] ", (conj.(transpose(vk))*Wk)[1], "\n")
        #     Hk[end, 1:end] = (conj.(transpose(vk))*Wk) # [1]
        # end
        
        # print("Julia eigvals ", eigsolve(Hk[1:i,1:i]), "\n")

        # # Compute the largest eigenpar (theta,s) of H_{k+1} 
        # # with the norm(s) = 1
        # eigvals, eigenvectors = eigs(Hk[1:i,1:i], which=:LM)
        # print("eigvals ", eigvals, "\n")
        # print("eigenvectors ", eigenvectors, "\n")

        print("hesse ", Hk, "\n")
        
        # Seans's code 
        julia_eig_solve =  eigen(Hk[1:i,1:i])
        julia_eigvals = julia_eig_solve.values
        julia_eigvects = julia_eig_solve.vectors
        # My code 
        # julia_eig_solve =  eigsolve(Hk[1:i,1:i])
        # julia_eigvals = julia_eig_solve[1]
        # julia_eigvects = julia_eig_solve[2]
        print("Julia eigvals ", julia_eigvals, "\n")
        print("Julia eigvectors ", julia_eigvects, "\n")
        theta = julia_eigvals[end]
        s =  julia_eigvects[:,end]/norm(julia_eigvects[:,end]) # Maximum eigvector
        print("theta", theta, "\n")
        print("s", s, "\n")


        # We only want the eigenvalue that is positive
        # position = 0

        # Problems with the code right now:
        # 1. I get a singular matrix when doing the direct solve. 
        # 2. Someting I get negative eigenvalues. 

        # max_eigvals = 0.0 
        # for i in eachindex(eigvals) # We need the eigenvalues to be positive, right (like before)?
        #     if eigvals[i] > 0 
        #         if eigvals[i] > max_eigvals
        #             max_eigvals = eigvals[i]
        #             theta = eigvals[i]
        #             position = i 
        #         end 
        #         print("position ", position, "\n")
        #     end 
        # end 
        # eigenvector = eigenvectors[:,position]
        # print("theta ", theta, "\n")
        # print("eigenvector ", eigenvector, "\n")
        # s = eigenvector/norm(eigenvector) # normalized_eigenvector
        # print("s ", s, "\n")
        
        # print("Vk ", Vk, "\n")
        u = Vk[:,1:i]*s # Compute the Ritz vector u  
        print("u ", u, "\n")
        u_hat = A*u # Should also have: A*u = W_k*s

        # u_hat test to see if A*u = W_k*s 
        u_hat_test = Wk[:,1:i]*s
        print("u_hat_test ", u_hat_test, "\n")

        # print("u_hat ", u_hat, "\n")
        # print("theta[1] ", real(theta[1]), "\n")
        # print("theta[1]*u ", theta[1]*u, "\n")
        r = theta[1]*u  .- u_hat # Residual vector 
        print("r ", r, "\n")
        
    end
        # print("norm of residual ", norm(r), "\n")
        # # if norm(r) <= tol
        # if real((conj.(transpose(r))*r)[1]) < tol
        #     print("Exited the loop using break")
        #     break 
        # end 

    # julia_eigvals = julia_eig_solve[1]
    # julia_eigvects = julia_eig_solve[2]
    print("julia_eigvals ", julia_eigvals, "\n")
    print("julia_eigvects ", julia_eigvects, "\n")
    print("Vk*conj.(transpose(Vk)) ", conj.(transpose(Vk))*Vk, "\n")

        # # Test 
        # print("julia_eigvects[:][:] ", julia_eigvects[:][:], "\n")
        # # eig_vect_matrix[:,:] = julia_eigvects[:][:]

        # eig_vect_matrix[:,1] = julia_eigvects[1][:]
        # eig_vect_matrix[:,2] = julia_eigvects[2][:]
        # eig_vect_matrix[:,3] = julia_eigvects[3][:]
        # print("eig_vect_matrix ", eig_vect_matrix, "\n")
        # A_test = Vk*Hk*conj.(transpose(Vk))

        # A_test = Vk*Hk*conj.(transpose(Vk))
        # print("A_test with Hk ", A_test, "\n")

        # A_test_A = Vk*A*conj.(transpose(Vk))
        # print("A_test_withA ", A_test_A, "\n")
        # print("Hk ", Hk, "\n")

        # print("A ", A, "\n")

        # Reset the different variables for when we run out of memory 
        # Vk = zeros(Float64, rows, cols)
        # # Vk = Array{Float64}(undef, rows, cols)
        # # Vk[1:end,1] = vk
        # Vk[1:end,1] = u

        # Wk = zeros(Float64, rows, cols)
        # # Wk = Array{Float64}(undef, rows, cols)
        # Wk[1:end,1] = u_hat
        
        # Hk = zeros(Float64, rows, cols)
        # # Hk = Array{Float64}(undef, rows, cols)
        # Hk[1,1] = real(theta[1])

        
        
    # end
    return real(theta),u
end 


A = [2.0 + im*0.0  -2.0 + im*0.0  0.0 + im*0.0;
-1.0 + im*0.0  3.0 + im*0.0  -1.0 + im*0.0;
0.0 + im*0.0  -1.0 + im*0.0  4.0 + im*0.0]
A[:,:] = (A .+ conj(transpose(A))) ./ 2
# A = Array{ComplexF64}(undef, 3, 3)
# A[1,1] = 2
# A[1,2] = -2
# A[1,3] = 0
# A[2,1] = -1
# A[2,2] = 3
# A[2,3] = -1 
# A[3,1] = 0
# A[3,2] = -1
# A[3,3] = 4
# A = A+conj.(transpose(A))
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
davidson_func_call = davidson_it(A)
print("Davidson max eigenvalue ", davidson_func_call[1], "\n")
print("Davidson max eigenvector ", davidson_func_call[2], "\n")
print("Julia eigenvalues ",eigen(A).values, "\n")

end 