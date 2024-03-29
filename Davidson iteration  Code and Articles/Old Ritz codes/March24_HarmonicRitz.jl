# module Davidson_HarmonizRitz_TestFile
using LinearAlgebra, Random, Arpack, KrylovKit # , bicgstab, cg

# @inline function projVec(dim::Integer, coeff, pVec::Vector{T}, sVec::Array{T})::Array{T} where T <: Number
@inline function projVec(coeff, pVec, sVec)
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
        print("((conj.(transpose(hk))*pk) ", (conj.(transpose(hk))*pk), "\n")
        print("conj.(transpose(hk))*fk) ", conj.(transpose(hk))*fk, "\n")
        coeff_first_proj = (real(conj.(transpose(hk))*pk)[1]/real(conj.(transpose(hk))*fk)[1])
        # coeff_first_proj = (real(conj.(transpose(hk))*pk)/real(conj.(transpose(hk))*fk))[1]
        print("coeff_first_proj ", coeff_first_proj, "\n")
        # print("fk ", fk, "\n")
        # print("pk ", pk, "\n")
        pkPrj = projVec(coeff_first_proj, fk, pk)
        # print("pkPrj ", pkPrj, "\n")
        A_proj = A * pkPrj .- (theta .* pkPrj)
        # print("A_proj ", A_proj, "\n")
        # coeff_second_proj = (real(conj.(transpose(hk))*A_proj)/real(conj.(transpose(hk))*fk))[1] 
        coeff_second_proj = (real(conj.(transpose(hk))*A_proj)[1]/real(conj.(transpose(hk))*fk)[1]) 
        # print("coeff_second_proj ", coeff_second_proj, "\n")
        vk = projVec(coeff_second_proj, fk, A_proj)

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
        # coeff_first_proj = ((conj.(transpose(hk))*b)/(conj.(transpose(hk))*fk))[1] 
        coeff_first_proj = ((conj.(transpose(hk))*b)/(conj.(transpose(hk))*fk))[1] 
        bPrj = projVec(coeff_first_proj, fk, b)
        A_proj = A * bPrj .- (theta .* bPrj)
        # coeff_second_proj = ((conj.(transpose(hk))*A_proj)/(conj.(transpose(hk))*fk))[1] 
        coeff_second_proj = ((conj.(transpose(hk))*A_proj)[1]/(conj.(transpose(hk))*fk)[1]) 
        t = projVec(coeff_second_proj, fk, A_proj)

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

function gramSchmidt!(i,wk,t,A,Wk,Vk,tol_a,tol_b)
    print("Vk ", Vk, "\n")
    print("Wk ", Wk, "\n")

    rnd = Array{ComplexF64}(undef,3,1) # Hard-coded for now 
    # nrm_a = norm(Wk[:,i])
    nrm_a = norm(wk)
    wk = wk/nrm_a
    # Wk[:,i] = Wk[:,i]/nrm_a
    # proj0 = conj.(transpose(Wk[:,1:i]))*Wk[:,i]
    proj0 = conj.(transpose(Wk[:,1:i]))*wk
    # wk_tilde = Wk[:,i] - Wk[:,1:i]*proj0
    wk_tilde = wk - Wk[:,1:i]*proj0
    print("wk_tilde ", wk_tilde, "\n")
    nrm_b = norm(wk_tilde)
    if nrm_b < tol_a
        t = t + rand!(rnd)
        # Wk[:,i] = A*t
        wk = A*t
        print("MGS \n")
        gramSchmidt!(i,wk,t,A,Wk,Vk,tol_a,tol_b)
    else 
        proj0 = proj0 + conj.(transpose(Wk[:,1:i]))*wk_tilde  # Wk[:,i]
        prj = (conj.(transpose(Wk[:,1:i]))*wk_tilde)/nrm_b
        # prj = (conj.(transpose(Wk[:,1:i]))*Wk[:,i])/nrm_b
    end 

    # print("norm(prj) ", norm(prj), "\n")
    while norm(prj) > tol_b 
        # print("while loop \n")
        wk_tilde = wk_tilde - Wk[:,1:i]*(conj.(transpose(Wk[:,1:i]))*wk_tilde)
        # print("wk_tilde ", wk_tilde, "\n")
        nrm_c = norm(wk_tilde)
        # print("nrm_c ", nrm_c, "\n")

        prj = (conj.(transpose(Wk[:,1:i]))*wk_tilde)/nrm_c
        proj0 = proj0 + (conj.(transpose(Wk[:,1:i]))*wk_tilde)
        nrm_b = nrm_b*nrm_c
        # print("norm(prj) ", norm(prj), "\n")
    end 
    t_tilde = t - Vk[:,1:i]*proj0
    print("t_tilde ", t_tilde, "\n")
    print("wk_tilde ", wk_tilde, "\n")
    Vk[:,i] = t_tilde/nrm_b
    Wk[:,i] = wk_tilde/nrm_b

    print("nrm_b ", nrm_b, "\n")
    # print("Vk[:,i] ", Vk[:,i], "\n")
    # print("Wk[:,i] ", Wk[:,i], "\n")
    print("Vk ", Vk, "\n")
    print("Wk ", Wk, "\n")

end


# function gramSchmidt!(i,t,A,Wk,Vk,tol_a,tol_b)
#     print("Vk ", Vk, "\n")
#     print("Wk ", Wk, "\n")

#     rnd = Array{ComplexF64}(undef,3,1) # Hard-coded for now 
#     nrm_a = norm(Wk[:,i])
#     wk = wk/norm(Wk[:,i])
#     # Wk[:,i] = Wk[:,i]/nrm_a
#     # proj0 = conj.(transpose(Wk[:,1:i]))*Wk[:,i]
#     proj0 = conj.(transpose(Wk[:,1:i]))*wk
#     # wk_tilde = Wk[:,i] - Wk[:,1:i]*proj0
#     wk_tilde = wk - Wk[:,1:i]*proj0
#     nrm_b = norm(wk_tilde)
#     if nrm_b < tol_a
#         t = t + rand!(rnd)
#         # Wk[:,i] = A*t
#         wk = A*t
#         print("MGS \n")
#         gramSchmidt!(i,t,A,Wk,Vk,tol_a,tol_b)
#     else 
#         proj0 = proj0 + conj.(transpose(Wk[:,1:i]))*wk_tilde  # Wk[:,i]
#         prj = (conj.(transpose(Wk[:,1:i]))*wk_tilde)/nrm_b
#         # prj = (conj.(transpose(Wk[:,1:i]))*Wk[:,i])/nrm_b
#     end 

#     # print("norm(prj) ", norm(prj), "\n")
#     while norm(prj) > tol_b 
#         # print("while loop \n")
#         wk_tilde = wk_tilde - Wk[:,1:i]*(conj.(transpose(Wk[:,1:i]))*wk_tilde)
#         nrm_c = norm(wk_tilde)
#         print("nrm_c ", nrm_c, "\n")

#         prj = (conj.(transpose(Wk[:,1:i]))*wk_tilde)/nrm_c
#         proj0 = proj0 + (conj.(transpose(Wk[:,1:i]))*wk_tilde)
#         nrm_b = nrm_b*nrm_c
#         # print("norm(prj) ", norm(prj), "\n")
#     end 
#     t_tilde = t - Vk[:,1:i]*proj0
#     print("t_tilde ", t_tilde, "\n")
#     print("wk_tilde ", wk_tilde, "\n")
#     Vk[:,i] = t_tilde/nrm_b
#     Wk[:,i] = wk_tilde/nrm_b

#     print("nrm_b ", nrm_b, "\n")
#     # print("Vk[:,i] ", Vk[:,i], "\n")
#     # print("Wk[:,i] ", Wk[:,i], "\n")
#     print("Vk ", Vk, "\n")
#     print("Wk ", Wk, "\n")

# end



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

    nrm_v = norm(v)
    vk = v/nrm_v # Vector 
    w = A*v # Vector 
    nrm_w = norm(w)
    wk = w/nrm_w # Vector 
    nrm_v = nrm_v
    # vk = vk/nrm_w # Vector 
    vk = vk/nrm_v # Vector 
    kk = (conj.(transpose(wk))*vk)[1] # Number
    Kk = zeros(ComplexF64, rows, cols)
    Wk = zeros(ComplexF64, rows, cols)
    Vk = zeros(ComplexF64, rows, cols)
    # print("kk ", kk, "\n")
    Kk[1,1] = kk
    Wk[:,1] = wk
    Vk[:,1] = vk
    theta_tilde = inv(kk)
    # print()
    r = wk - theta_tilde*vk

    fk = vk # Vector 
    hk = wk # Vector 

    # Test matrix to see if 
    # conj.(transpose(eig_vect_matrix))*Hk_hat*eig_vect_matrix = A
    eig_vect_matrix = zeros(ComplexF64, rows, cols)
    julia_eigvals = 0
    julia_eigvects = 0

    # Part 2: Inner loop 
    for i = 2:cols # Iterate through all of the columns 
        print("i ", i, "\n")

        # Solve for t using bicgstab 
        # 1. Here we using only the diagonal of A and we make the 
        # last element of u_tilde and u_hat equal to 0. 
        # t = bicgstab_matrix_ritz(((I-(z_mod*conj.(transpose(z_mod))))*(A-
        # real(theta_tilde[1])*I)*(I-Vk*s*conj.(transpose(z_mod))*A)),theta_tilde,z,
        # theta_tilde*wk)
        
        # Not sure about this... Ask Sean about how his changes to my bicgstab
        # work. 
        # bicgstab_matrix_ritz(A, theta, fk, hk, b)
        t = bicgstab_matrix_ritz(A, theta_tilde, fk, hk, r)

        wk = A*t # Vector 
        # We don't do the life below for the wk version of the code (see MGS)
        Wk[:,i] = wk # W_{k+1} # Added wk to Wk version 

        # MGS (TBD for tolerance values)
        tol_a = 1e-2
        tol_b = 1e-3

        gramSchmidt!(i-1,wk,t,A,Wk[:,1:i],Vk[:,1:i],tol_a,tol_b) # wk version 
        # gramSchmidt!(i,t,A,Wk,Vk,tol_a,tol_b) # Added wk to Wk version 
        # gramSchmidt!(Vk, i) # Sean's code 

        # Modifies Vk and Wk 
        # print("1:2 ", 1:2, "\n")
        # print("Kk[i+1,1:i] ", Kk[i+1,1:i], "\n")
        # print("conj.(transpose(Wk[:,i]))*Vk[:,1:i-1] ", conj.(transpose(Wk[:,i]))*Vk[:,1:i], "\n")
        
        # Kk[i+1,1:i] = conj.(transpose(Wk[:,i]))*Vk[:,1:i]
        Kk[i,1:i-1] = conj.(transpose(Wk[:,i]))*Vk[:,1:i-1]
        # Kk[1:i,i+1] = conj.(transpose(Wk[:,1:i]))*Vk[:,i]
        Kk[1:i-1,i] = conj.(transpose(Wk[:,1:i-1]))*Vk[:,i]
        # print("conj.(transpose(Wk[:,i])) ", conj.(transpose(Wk[:,i])), "\n")
        # print("Vk[:,i]", Vk[:,i], "\n")
        Kk[i-1,i-1] = (conj.(transpose(Wk[:,i]))*Vk[:,i])[1]

        # Compute the smallest eigenpair of Kk 
        print("Kk ", Kk[1:i,1:i], "\n")
        print("size(Kk) ", size(Kk), "\n")
        julia_eig_solve =  eigsolve(Kk[1:i,1:i])
        julia_eigvals = julia_eig_solve[1]
        julia_eigvects = julia_eig_solve[2]
        print("julia_eigvects ", julia_eigvects, "\n")

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
        print("s ", s, "\n")

        fk = Vk[:,1:i]*s
        # nrm_fk = norm(fk)
        hk = (Wk[:,1:i]*s)/norm(fk) # Harmonic Ritz vector 
        r = hk - theta_tilde*fk


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