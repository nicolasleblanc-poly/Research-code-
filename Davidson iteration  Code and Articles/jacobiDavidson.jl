using LinearAlgebra, Random 

@inline function projVec(dim::Integer, pVec::Vector{T}, sVec::Array{T})::Array{T} where T <: Number

	return sVec .- (BLAS.dotc(dim, pVec, 1, sVec, 1) .* pVec)
end

# This is a biconjugate gradient program without a preconditioner. 
# m is the maximum number of iterations
function bad_bicgstab_matrix(A, theta, u, b)
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
        pkPrj = projVec(dim, u, pk)
        vk = projVec(dim, u, A * pkPrj .- (theta .* pkPrj))
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
        bPrj = projVec(dim, u, b) 
        t = projVec(dim, u, A * bPrj .- (theta .* bPrj))
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

function gramSchmidt!(basis::Array{T}, n::Integer, tol::Float64) where T <: Number

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
		gramSchmidt!(basis, n, tol)		
	else
		# renormalize orthogonalized vector
		basis[:,n] = basis[:,n] ./ nrm
	end 
end

function jacDavRitz(basis::Array{ComplexF64}, hesse::Array{ComplexF64}, 
	opt::Array{ComplexF64}, vecDim::Integer, repDim::Integer, tol::Float64)::Tuple 

	print("opt ", opt, "\n")
	### memory initialization
	# basis is Vk 
	outVec = Vector{ComplexF64}(undef, vecDim) # Wk 
	resVec = Vector{ComplexF64}(undef, vecDim)
	ritzVec = Vector{ComplexF64}(undef, vecDim)
	# set starting vector
	# rand!(view(basis, :, 1))
	basis[:,1] = [0.9026710530499227 + 0.8801240709989707im;
    0.8797904471904741 + 0.13830203407554031im; 
    0.3804996099871757 + 0.19827553581504442im]
	print("view(basis, :, 1) ", view(basis, :, 1), "\n")
	# normalize starting vector
	nrm = BLAS.nrm2(vecDim, view(basis,:,1), 1)
	basis[:, 1] = basis[:, 1] ./ nrm
	### algorithm initialization
	outVec = opt * basis[:, 1] 
	print("outVec ", outVec, "\n")
	# Hessenberg matrix
	hesse[1,1] = BLAS.dotc(vecDim, view(basis, :, 1), 1, outVec, 1) 
	# Ritz value
	theta = hesse[1,1] 

	print("theta ", theta, "\n")
	
	# Ritz vector
	ritzVec[:] = basis[:, 1]
	# Negative residual vector
	resVec = (theta .* ritzVec) .- outVec
	print("resVec ", resVec, "\n")

	for itr in 2 : repDim
		print("opt check ", opt, "\n")
		print("theta check ", theta, "\n")
        print("ritzVec check ", ritzVec, "\n")
        print("resVec check ", resVec, "\n")

		# Jacobi-Davidson direction
		basis[:, itr] = bad_bicgstab_matrix(opt, theta, ritzVec, resVec)
		
		print("tsolve ", basis[:, itr], "\n")
		
		# orthogonalize
		gramSchmidt!(basis, itr, tol)
		# new image
		outVec = opt * basis[:, itr] 
		# update Hessenberg
		hesse[1 : itr, itr] = BLAS.gemv('C', view(basis, :, 1 : itr), outVec)
		hesse[itr, 1 : (itr - 1)] = conj(hesse[1 : (itr - 1), itr])
		
		print("hesse ", hesse, "\n")
		
		# eigenvalue decomposition, largest real eigenvalue last. 
		# should replace by BLAS operation
		eigSys = eigen(view(hesse, 1 : itr, 1 : itr)) 
		
		print("eig_Sys[:,end] ", eigSys.vectors[:,end], "\n")
		
		# update Ritz vector
		print("eigSys.values ", eigSys.values, "\n")
		theta = eigSys.values[end]
		print("theta ", theta, "\n")
		ritzVec[:] = basis[:, 1 : itr] * (eigSys.vectors[:, end])
		outVec = opt * ritzVec
		print("ritzVec[:,end] ", ritzVec[:,end], "\n")
		# update residual vector
		resVec = (theta * ritzVec) .- outVec
		print("resVec ", resVec, "\n")
		# add tolerance check here
	end
	print("conj(transpose(basis))*basis ",conj(transpose(basis))*basis,"\n")

	return (real(theta), ritzVec[:,end])
end
# test
opt = [2.0 + im*0.0  -2.0 + im*0.0  0.0 + im*0.0;
	-1.0 + im*0.0  3.0 + im*0.0  -1.0 + im*0.0;
	0.0 + im*0.0  -1.0 + im*0.0  4.0 + im*0.0]

opt[:,:] = (opt .+ conj(transpose(opt))) ./ 2

dims = size(opt)

basis = Array{ComplexF64}(undef, dims[1], dims[2])
hesse = zeros(ComplexF64, dims[2], dims[2])

(val, vec) = jacDavRitz(basis, hesse, opt, dims[1], dims[2], 1.0e-6)
print("val ", val, "\n")
print("vec ", vec, "\n")

trueEig = eigen(opt) 