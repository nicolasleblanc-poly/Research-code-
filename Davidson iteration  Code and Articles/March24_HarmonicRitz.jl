# module Davidson_HarmonizRitz_TestFile
using LinearAlgebra, Random, Arpack, KrylovKit # , bicgstab, cg

@inline function projVec(dim::Integer, coeff::Integer, pVec::Vector{T}, sVec::Array{T})::Array{T} where T <: Number
    return sVec .- (coeff .* pVec)
	# return sVec .- (BLAS.dotc(dim, pVec, 1, sVec, 1) .* pVec) # Sean old code 
end

function bicgstab_matrix_ritz(A, theta, fk, hk, b)
	dim = size(A)[1]
    v_m1 = p_m1 = xk_m1 = zeros(ComplexF64,length(b),1)
    # tol = 1e-4
    # Ax=0 since the initial xk is 0
    r0 = r_m1 = b 
    rho_m1 = alpha = omega_m1 = 1
    # for k in 1:length(b)
    # we don't want a perfect solve, should fix this though
    for k in 1 : 2
        rho_k = conj.(transpose(r0))*r_m1  
        # beta calculation
        # First term 
        first_term = rho_k/rho_m1
        # Second term 
        second_term = alpha/omega_m1
        # Calculation 
        beta = first_term*second_term
        pk = r_m1 .+ beta.*(p_m1-omega_m1.*v_m1)

        # Projections 
        coeff_first_proj = (conj.(transpose(hk))*pk)/(conj.(transpose(hk))*fk) 
        pkPrj = projVec(dim, coeff_first_proj, fk, pk)
        A_proj = A * pkPrj .- (theta .* pkPrj)
        coeff_second_proj = (conj.(transpose(hk))*A_proj)/(conj.(transpose(hk))*fk) 
        vk = projVec(dim, coeff_second_proj, u, A_proj)

        # Sean old code for the projections above
        # pkPrj = projVec(dim, u, pk)
        # vk = projVec(dim, u, A * pkPrj .- (theta .* pkPrj))


        # (l[1])*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, pk)
        # alpha calculation
        # Bottom term
        bottom_term = conj.(transpose(r0))*vk
        # Calculation 
        alpha = rho_k / bottom_term 
        
        h = xk_m1 + alpha.*pk
        # If h is accurate enough, then set xk=h and quantity
        # What does accurate enough mean?
        s = r_m1 - alpha.*vk

        # Projections 
        coeff_first_proj = (conj.(transpose(hk))*b)/(conj.(transpose(hk))*fk) 
        bPrj = projVec(dim, coeff_first_proj, fk, b)
        A_proj = A * bPrj .- (theta .* bPrj)
        coeff_second_proj = (conj.(transpose(hk))*A_proj)/(conj.(transpose(hk))*fk) 
        t = projVec(dim, coeff_second_proj, u, A_proj)

        # Old Sean code for the projections above 
        # bPrj = projVec(dim, u, b) 
        # t = projVec(dim, u, A * bPrj .- (theta .* bPrj))


        # (l[1])*asym_vect(gMemSlfN,gMemSlfA, cellsA, chi_inv_coeff, P, s)
        # omega_k calculation 
        # Top term 
        ts = conj.(transpose(t))*s
        # Bottom term
        tt = conj.(transpose(t))*t
        # Calculation 
        omega_k = ts ./ tt
        xk_m1 = h + omega_k.*s
        r_old = r_m1
        r_m1 = s-omega_k.*t
        # print("conj.(transpose(r_m1))*r_m1  ", conj.(transpose(r_m1))*r_m1 , "\n")
        # if real((conj.(transpose(r_m1))*r_m1)[1]) < tol
        # # if norm(r_m1)-norm(r_old) < tol
        #     print("bicgstab break \n")
        #     print("real((conj.(transpose(r_m1))*r_m1)[1])",real((conj.(transpose(r_m1))*r_m1)[1]),"\n")
        #     break 
        # end 
    end
    return xk_m1
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
    tol = 1e-6 # Tolerance for which the program will converge 

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
    

    Wk_tilde = zeros(ComplexF64, rows, cols)
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
    theta_tilde = hk # Number 
    r = u_hat - real(theta_tilde)*u_tilde # Vector
    # print("r ", r, "\n")

    t = zeros(ComplexF64, rows, 1)
    s = zeros(ComplexF64, rows, 1)
    rand!(s)

    z = A*u_tilde

    # Test matrix to see if 
    # conj.(transpose(eig_vect_matrix))*Hk_hat*eig_vect_matrix = A
    eig_vect_matrix = zeros(ComplexF64, rows, cols)
    julia_eigvals = 0
    julia_eigvects = 0

    # Part 2: Inner loop 
    for i = 2:cols # Iterate through all of the columns 
        print("i ", i, "\n")

        # diagonal_A = diag(A)
        # A_diagonal_matrix = Diagonal(diagonal_A)
        # print("A_diagonal_matrix ", A_diagonal_matrix, "\n")

        # u_tilde_mod = copy(u_tilde)
        # # print("u_mod[end:end,1] ", u_mod[end:end,1], "\n")
        # u_tilde_mod[end] = 0.0

        # u_hat_mod = copy(u_hat)
        # # print("u_mod[end:end,1] ", u_mod[end:end,1], "\n")
        # u_hat_mod[end] = 0.0

        z_mod = copy(z)
        z_mod[end] = 0.0

        # Solve for t using bicgstab 
        # 1. Here we using only the diagonal of A and we make the 
        # last element of u_tilde and u_hat equal to 0. 
        # t = bicgstab_matrix_ritz(((I-(z_mod*conj.(transpose(z_mod))))*(A-
        # real(theta_tilde[1])*I)*(I-Vk*s*conj.(transpose(z_mod))*A)),theta_tilde,z,
        # theta_tilde*wk)
        
        # Not sure about this... Ask Sean about how his changes to my bicgstab
        # work. 
        t = bicgstab_matrix_ritz(((I-(z_mod*conj.(transpose(z_mod))))*(A-
        real(theta_tilde[1])*I)*(I-Vk*s*conj.(transpose(z_mod))*A)),theta_tilde,z,
        theta_tilde*wk)
        # bicgstab_matrix_ritz(A, theta, u, b)

        # MGS (TBD)
        # gramSchmidt!(Vk, i) # Sean's code 

        # t_tilde = t - Vk[:,1:i-1]*inv(Lk[1:i-1,1:i-1])*
        # conj.(transpose(Wk[:,1:i-1]))*t
        # # print("t_tilde ", t_tilde, "\n")
        # vk = t_tilde/norm(t_tilde) # v_{k+1}
        # Vk[:,i] = vk # Add the new vk to the Vk matrix -> V_{k+1}

        # New wk that will be added as a new column to Wk
        wk = A*t # w_{k+1} = A*v_{k+1}
        # print("wk ", wk, "\n")
        # print("Wk ", Wk, "\n")
        
        # print("new Wk ", Wk, "\n")

        wk_tilde = wk - Wk*conj.(transpose(Wk))*wk
        # Expand W_k with this vector to W_{k+1}
        Wk[:,i] = wk_tilde/norm(wk_tilde) # W_{k+1}

        t_tilde = t - Vk*conj.(transpose(Wk))*wk
        vk = t_tilde/norm(wk_tilde)
        Vk[:,i] = vk # Add the new vk to the Vk matrix -> V_{k+1}
        Hk_tilde = inv(conj.(transpose(Wk)*Vk))
        # Hk_tilde = (W_k*Vk)^{-1}
        # It is not necessary to invert W_k*Vk since the harmonic Ritz values
        # are simply the inverses of the eigenvalues of W_k*Vk.

        # lk = conj.(transpose(wk))*Vk[:,1:i]
        # # print("lk ", lk, "\n")
        # # print("Lk ", Lk, "\n")
        # # Expand L_k with this vector to W_{k+1}
        # Lk[i,1:i] = lk # Row of matrix -> L_{k+1}

        # hk = conj.(transpose(wk))*Wk[:,1:i]
        # # H_hat_{k+1}
        # Hk_hat[i,1:i] = hk # Row of matrix
        # Hk_hat[1:i,i] = conj.(transpose(hk)) # Column of matrix
        # # print("Hk_hat ", Hk_hat, "\n")

        # Hk_tilde = inv(Lk[1:i,1:i])*Hk_hat[1:i,1:i] # H_tilde_{k+1}

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
        # Normalized minimum eigvector
        s =  julia_eigvects[position][:]/norm(julia_eigvects[position][:]) 

        # print("Julia eigvals ", julia_eigvals, "\n")
        # print("Julia eigvectors ", julia_eigvects, "\n")
        # theta_tilde = julia_eigvals[end]
        # s =  julia_eigvects[end][:] # Minimum eigvector
        print("theta_tilde ", theta_tilde, "\n")
        # print("s ", s, "\n")
        
        # Compute the harmonic Ritz vector 
        u_tilde = (Vk[:,1:i]*s)/norm(Vk[:,1:i]*s)
        u_hat = A*u_tilde 
        print("u_hat ", u_hat, "\n")
        print("u_hat_test ", (Wk[1:i,1:i]*s)/(norm(Vk[1:i,1:i]*s)), "\n")
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
        # print("conj(transpose(Vk))*Vk ", conj(transpose(Vk))*Vk, "\n")
        
    end
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
# A = [2.0 + im*0.0  -2.0 + im*0.0  0.0 + im*0.0;
# 	-1.0 + im*0.0  3.0 + im*0.0  -1.0 + im*0.0;
# 	0.0 + im*0.0  -1.0 + im*0.0  4.0 + im*0.0]
A = (A+conj.(transpose(A)))/2
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

print("Davidson minimum positive eigenvalue ",davidson_it(A), "\n")
print("Julia direct solve ",eigen(A).values, "\n")
# end 