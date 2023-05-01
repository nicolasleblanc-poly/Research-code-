using LinearAlgebra, Random
function jacDavRitzHarm_basic(trgBasis::Array{ComplexF64}, srcBasis::Array{ComplexF64}, kMat::Array{ComplexF64}, opt::Array{ComplexF64}, vecDim::Integer, repDim::Integer, loopDim::Integer,tol::Float64)::Float64
	### memory initialization
	resVec = Vector{ComplexF64}(undef, vecDim)
	hRitzTrg = Vector{ComplexF64}(undef, vecDim)
	hRitzSrc = Vector{ComplexF64}(undef, vecDim)
	bCoeffs1 = Vector{ComplexF64}(undef, repDim)
	bCoeffs2 = Vector{ComplexF64}(undef, repDim)
	# set starting vector
	rand!(view(srcBasis, :, 1)) # vk
	# normalize starting vector
	nrm = BLAS.nrm2(vecDim, view(srcBasis,:,1), 1) # norm(vk)
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	### algorithm initialization
	trgBasis[:, 1] = opt * srcBasis[:, 1] # Wk
	nrm = BLAS.nrm2(vecDim, view(trgBasis,:,1), 1)
	trgBasis[:, 1] = trgBasis[:, 1] ./ nrm # Wk
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	# representation of opt^{-1} in trgBasis
	kMat[1,1] = BLAS.dotc(vecDim, view(trgBasis, :, 1), 1,
		view(srcBasis, :, 1), 1) # Kk
	# Ritz value
	eigPos = 1
	theta = 1 / kMat[1,1] # eigenvalue 
	# Ritz vectors
	hRitzTrg[:] = trgBasis[:, 1] # hk = wk 
	hRitzSrc[:] = srcBasis[:, 1] # fk = vk
	# Negative residual vector
	resVec = (theta .* hRitzSrc) .- hRitzTrg # theta_tilde*vk - wk

	# Code for if we just want the inner loop, so with no restart  
	for itr in 2 : repDim # Need to determine when this for loops stops 
		# depending on how much memory the laptop can take before crashing.
		prjCoeff = BLAS.dotc(vecDim, hRitzTrg, 1, hRitzSrc, 1)
		# calculate Jacobi-Davidson direction
		srcBasis[:, itr] = bad_bicgstab_matrix(opt, theta, hRitzTrg,
			hRitzSrc, prjCoeff, resVec)
		trgBasis[:, itr] = opt * srcBasis[:, itr]
		# orthogonalize
		gramSchmidtHarm!(trgBasis, srcBasis, bCoeffs1, bCoeffs2, opt,
			itr, tol)
		# update inverse representation of opt^{-1} in trgBasis
		kMat[1 : itr, itr] = BLAS.gemv('C', view(trgBasis, :, 1 : itr),
			view(srcBasis, :, itr))
		# assuming opt^{-1} Hermitian matrix
		kMat[itr, 1 : (itr - 1)] = conj(kMat[1 : (itr-1), itr])
		# eigenvalue decomposition, largest real eigenvalue last.
		# should replace by BLAS operation
		eigSys = eigen(view(kMat, 1 : itr, 1 : itr))
	

		# A better way (in my honest opinion) would be to find the largest
		# positive eigenvalue and then just do 1/it since that what we want.
		# Let's find the largest positive eigenvalue found using the Julia 
		# solver on kMat done above.
		global max_eigval = 0
		position = 1
		for i in eachindex(eigSys.values)
			if max_eigval < real(eigSys.values[i]) 
				theta_tilde = real(eigSys.values[i])
				position = i 
				global max_eigval = real(theta_tilde)
				# print("position ", position, "\n")
				# print("Updated position ", i, " times \n")
			end 
			# print("i ", i, "\n")
			# print("min_eigval ", min_eigval, "\n")
		end 
		theta = 1/max_eigval

		# update residual vector
		resVec = (theta * hRitzSrc) .- hRitzTrg
 
		# add tolerance check here
		if norm(resVec) < tol
			print("Converged off tolerance \n")
			return real(theta) 
			# println(real(theta))
		end
		print("norm(resVec) basic program ", norm(resVec),"\n")
	end
 
	print("Didn't converge off tolerance for basic program. 
		Atteined max set number of iterations \n")
	return real(theta)
end


function jacDavRitzHarm_restart(trgBasis::Array{ComplexF64}, 
	srcBasis::Array{ComplexF64}, kMat::Array{ComplexF64}, 
	opt::Array{ComplexF64}, vecDim::Integer, repDim::Integer, 
	restartDim::Integer,innerLoopDim::Integer,nb_rest_keep_vals::Integer,
	tol::Float64)::Float64

	restart_resVec = Vector{ComplexF64}(undef, vecDim)
	restart_hRitzTrg = Vector{ComplexF64}(undef, vecDim)
	restart_hRitzSrc = Vector{ComplexF64}(undef, vecDim)
	restart_trgBasis = Array{ComplexF64}(undef, vecDim, vecDim)
	restart_srcBasis = Array{ComplexF64}(undef, vecDim, vecDim)
	restart_kMat = zeros(ComplexF64, vecDim, vecDim)
	restart_theta = 0 

	### memory initialization
	resVec = Vector{ComplexF64}(undef, vecDim)
	hRitzTrg = Vector{ComplexF64}(undef, vecDim)
	hRitzSrc = Vector{ComplexF64}(undef, vecDim)
	bCoeffs1 = Vector{ComplexF64}(undef, repDim)
	bCoeffs2 = Vector{ComplexF64}(undef, repDim)
	# set starting vector
	rand!(view(srcBasis, :, 1)) # vk
	# normalize starting vector
	nrm = BLAS.nrm2(vecDim, view(srcBasis,:,1), 1) # norm(vk)
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	### algorithm initialization
	trgBasis[:, 1] = opt * srcBasis[:, 1] # Wk
	nrm = BLAS.nrm2(vecDim, view(trgBasis,:,1), 1)
	trgBasis[:, 1] = trgBasis[:, 1] ./ nrm # Wk
	srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
	# representation of opt^{-1} in trgBasis
	kMat[1,1] = BLAS.dotc(vecDim, view(trgBasis, :, 1), 1,
		view(srcBasis, :, 1), 1) # Kk
	# Ritz value
	eigPos = 1
	theta = 1 / kMat[1,1] # eigenvalue 
	# Ritz vectors
	hRitzTrg[:] = trgBasis[:, 1] # hk = wk 
	hRitzSrc[:] = srcBasis[:, 1] # fk = vk
	# Negative residual vector
	resVec = (theta .* hRitzSrc) .- hRitzTrg # theta_tilde*vk - wk

	# innerLoopDim = Int(repDim/4)

	# Code with restart
	# Outer loop
	for it in 1:restartDim # Need to think this over 
		# Inner loop
		print("it ", it, "\n")
		if it > 1
			# Before we restart, we will create a new version of everything 
			# Note: vecDim = dims[1] = dims[2] here. 
			resVec = restart_resVec
			
			hRitzTrg = restart_hRitzTrg
			
			hRitzSrc = restart_hRitzSrc

			bCoeffs1 = Vector{ComplexF64}(undef, repDim)
			bCoeffs2 = Vector{ComplexF64}(undef, repDim)
			
			trgBasis = Array{ComplexF64}(undef, vecDim, vecDim)
			trgBasis[:,innerLoopDim-nb_rest_keep_vals+1:innerLoopDim] = restart_trgBasis
			
			srcBasis = Array{ComplexF64}(undef, vecDim, vecDim)
			srcBasis[:,1:nb_rest_keep_vals] = restart_srcBasis
			print("restart_srcBasis ", restart_srcBasis,"\n")

			theta = restart_theta 

			kMat = zeros(ComplexF64, vecDim, vecDim)
			kMat[1:nb_rest_keep_vals,1:nb_rest_keep_vals] = restart_kMat 

			# # normalize starting vector
			# nrm = BLAS.nrm2(vecDim, view(srcBasis,:,1), 1) # norm(vk)
			# srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
			# ### algorithm initialization
			# trgBasis[:, 1] = opt * srcBasis[:, 1] # Wk
			# nrm = BLAS.nrm2(vecDim, view(trgBasis,:,1), 1)
			# trgBasis[:, 1] = trgBasis[:, 1] ./ nrm # Wk
			# srcBasis[:, 1] = srcBasis[:, 1] ./ nrm # Vk
			# # representation of opt^{-1} in trgBasis
			# kMat[1,1] = BLAS.dotc(vecDim, view(trgBasis, :, 1), 1,
			# 	view(srcBasis, :, 1), 1) # Kk
			# # Ritz value
			# eigPos = 1
			# theta = 1 / kMat[1,1] # eigenvalue 
			# # Ritz vectors
			# hRitzTrg[:] = trgBasis[:, 1] # hk = wk 
			# hRitzSrc[:] = srcBasis[:, 1] # fk = vk
			# # Negative residual vector
			# resVec = (theta .* hRitzSrc) .- hRitzTrg # theta_tilde*vk - wk

			# The key different with the for loop for it = 0 is that here we 
			# are starting at nb_rest_keep_vals and not 2.
			for itr in 1 : innerLoopDim # Need to determine when this for loops stops 
				# depending on how much memory the laptop can take before crashing.
				prjCoeff = BLAS.dotc(vecDim, hRitzTrg, 1, hRitzSrc, 1)
				# calculate Jacobi-Davidson direction
				srcBasis[:, nb_rest_keep_vals+itr] = bad_bicgstab_matrix(opt, theta, hRitzTrg,
					hRitzSrc, prjCoeff, resVec)
				trgBasis[:, nb_rest_keep_vals+itr] = opt * srcBasis[:, nb_rest_keep_vals+itr]
				# orthogonalize
				gramSchmidtHarm!(trgBasis, srcBasis, bCoeffs1, bCoeffs2, opt,
					nb_rest_keep_vals+itr, tol)

				print("kMat 1 when it>2", kMat, "\n")	

				# update inverse representation of opt^{-1} in trgBasis
				kMat[1 : itr, itr] = BLAS.gemv('C', 
				view(trgBasis, :, 1 : itr),
					view(srcBasis, :, itr))
				# assuming opt^{-1} Hermitian matrix
				kMat[itr, 1 : (itr - 1)] = conj(kMat[1 : (itr-1), itr])
				# eigenvalue decomposition, largest real eigenvalue last.
				# should replace by BLAS operation

				print("kMat 2 when it>2 ", kMat, "\n")

				eigSys = eigen(view(kMat, 1 : itr, 1 : itr))
		
				global max_eigval = 0
				position = 1
				for i in eachindex(eigSys.values)
					if max_eigval < real(eigSys.values[i]) 
						theta_tilde = real(eigSys.values[i])
						position = i 
						global max_eigval = real(theta_tilde)
						# print("position ", position, "\n")
						# print("Updated position ", i, " times \n")
					end 
					# print("i ", i, "\n")
					# print("min_eigval ", min_eigval, "\n")
				end 
				theta = 1/max_eigval
		
	
				# update residual vector
				resVec = (theta * hRitzSrc) .- hRitzTrg
		 
				# add tolerance check here
				if norm(resVec) < tol
					print("Converged off tolerance \n")
					return real(theta) 
					# println(real(theta))
				end
				print("norm(resVec) restart program ", norm(resVec),"\n")
			end
			print(itr)
			restart_resVec = resVec
			restart_hRitzTrg = trgBasis[:,innerLoopDim]
			restart_hRitzSrc = srcBasis[:,innerLoopDim]
			restart_trgBasis = trgBasis[:,innerLoopDim-nb_rest_keep_vals+1:innerLoopDim]
			print("innerLoopDim-nb_rest_keep_vals ", innerLoopDim-nb_rest_keep_vals, "\n")
			print("innerLoopDim ", innerLoopDim, "\n")
			restart_srcBasis = srcBasis[:,innerLoopDim-nb_rest_keep_vals+1:innerLoopDim]
			print("restart_srcBasis ", restart_srcBasis, "\n")
			# restart_kMat = kMat[:,innerLoopDim-nb_rest_keep_vals+1:innerLoopDim]
			restart_kMat = kMat[1:innerLoopDim-nb_rest_keep_vals,1:innerLoopDim-nb_rest_keep_vals]
			print("restart_kMat for it>2 ", restart_kMat, "\n")
			restart_theta = theta

			println("Finished inner loop 2 \n")
	
			# Once we have ran out of memory, we want to restart the inner loop 
			# but not with random starting vectors and matrices but with the ones
			# from the last inner loop iteration that was done before running out 
			# of memory. 

		# Essentially if it == 0
		else
			for itr in 2 : innerLoopDim # Need to determine when this for loops stops 
				# depending on how much memory the laptop can take before crashing.
				prjCoeff = BLAS.dotc(vecDim, hRitzTrg, 1, hRitzSrc, 1)
				# calculate Jacobi-Davidson direction
				srcBasis[:, itr] = bad_bicgstab_matrix(opt, theta, hRitzTrg,
					hRitzSrc, prjCoeff, resVec)
				trgBasis[:, itr] = opt * srcBasis[:, itr]
				# orthogonalize
				gramSchmidtHarm!(trgBasis, srcBasis, bCoeffs1, bCoeffs2, opt,
					itr, tol)

				print("kMat 1 when it=1 ", kMat, "\n")
				# update inverse representation of opt^{-1} in trgBasis
				kMat[1 : itr, itr] = BLAS.gemv('C', view(trgBasis, :, 1 : itr),
					view(srcBasis, :, itr))
				# assuming opt^{-1} Hermitian matrix
				kMat[itr, 1 : (itr - 1)] = conj(kMat[1 : (itr-1), itr])
				print("kMat 2 when it=1 ", kMat, "\n")

				# eigenvalue decomposition, largest real eigenvalue last.
				# should replace by BLAS operation
				eigSys = eigen(view(kMat, 1 : itr, 1 : itr))
		
				global max_eigval = 0
				position = 1
				for i in eachindex(eigSys.values)
					if max_eigval < real(eigSys.values[i]) 
						theta_tilde = real(eigSys.values[i])
						position = i 
						global max_eigval = real(theta_tilde)
						# print("position ", position, "\n")
						# print("Updated position ", i, " times \n")
					end 
					# print("i ", i, "\n")
					# print("min_eigval ", min_eigval, "\n")
				end 
				theta = 1/max_eigval
		

				# update residual vector
				resVec = (theta * hRitzSrc) .- hRitzTrg
		
				# add tolerance check here
				if norm(resVec) < tol
					print("Converged off tolerance \n")
					return real(theta) 
					# println(real(theta))
				end
				print("norm(resVec) restart program ", norm(resVec),"\n")
			end
			# print("full srcBasis ", srcBasis, "\n")


			restart_resVec = resVec
			restart_hRitzTrg = trgBasis[:,innerLoopDim]
			restart_hRitzSrc = srcBasis[:,innerLoopDim]
			restart_trgBasis = trgBasis[:,innerLoopDim-nb_rest_keep_vals+1:innerLoopDim]
			print("innerLoopDim-nb_rest_keep_vals ", innerLoopDim-nb_rest_keep_vals, "\n")
			print("innerLoopDim ", innerLoopDim, "\n")
			restart_srcBasis = srcBasis[:,innerLoopDim-nb_rest_keep_vals+1:innerLoopDim]
			print("restart_srcBasis ", restart_srcBasis, "\n")
			# restart_kMat = kMat[:,innerLoopDim-nb_rest_keep_vals+1:innerLoopDim]
			restart_kMat = kMat[1:innerLoopDim-nb_rest_keep_vals,1:innerLoopDim-nb_rest_keep_vals]
			print("restart_kMat for it = 1 ", restart_kMat, "\n")
			restart_theta = theta

			println("Finished inner loop 1 \n")

			# Once we have ran out of memory, we want to restart the inner loop 
			# but not with random starting vectors and matrices but with the ones
			# from the last inner loop iteration that was done before running out 
			# of memory. 
		end
		
	end 
	print("Didn't converge off tolerance for restart program. 
		Atteined max set number of iterations \n")
	return real(theta)
end

# perform Gram-Schmidt on target basis, adjusting source basis accordingly
function gramSchmidtHarm!(trgBasis::Array{T}, srcBasis::Array{T},
	bCoeffs1::Vector{T}, bCoeffs2::Vector{T}, opt::Array{T}, n::Integer,
	tol::Float64) where T <: Number
	# dimension of vector space
	dim = size(trgBasis)[1]
	# initialize projection norm
	prjNrm = 1.0
	# initialize projection coefficient memory
	bCoeffs1[1:(n-1)] .= 0.0 + im*0.0
	# check that basis does not exceed dimension
	if n > dim
		error("Requested basis size exceeds dimension of vector space.")
	end
	# norm of proposed vector
	nrm = BLAS.nrm2(dim, view(trgBasis,:,n), 1)
	# renormalize new vector
	trgBasis[:,n] = trgBasis[:,n] ./ nrm
	srcBasis[:,n] = srcBasis[:,n] ./ nrm
	# guarded orthogonalization
	while prjNrm > (tol * 100) && abs(nrm) > tol
		### remove projection into existing basis
 		# calculate projection coefficients
 		BLAS.gemv!('C', 1.0 + im*0.0, view(trgBasis, :, 1:(n-1)),
 			view(trgBasis, :, n), 0.0 + im*0.0,
 			view(bCoeffs2, 1:(n -1)))
 		# remove projection coefficients
 		BLAS.gemv!('N', -1.0 + im*0.0, view(trgBasis, :, 1:(n-1)),
 			view(bCoeffs2, 1:(n -1)), 1.0 + im*0.0,
 			view(trgBasis, :, n))
 		# update total projection coefficients
 		bCoeffs1 .= bCoeffs2 .+ bCoeffs1
 		# calculate projection norm
 		prjNrm = BLAS.nrm2(n-1, bCoeffs2, 1)
 	end
 	# remaining norm after removing projections
 	nrm = BLAS.nrm2(dim, view(trgBasis,:,n), 1)
	# check that remaining vector is sufficiently large
	if abs(nrm) < tol
		# switch to random search direction
		rand!(view(srcBasis, :, n))
		trgBasis[:, n] = opt * srcBasis[:, n]
		gramSchmidtHarm!(trgBasis, srcBasis, bCoeffs1, bCoeffs2,
			opt, n, tol)
	else
		# renormalize
		trgBasis[:,n] = trgBasis[:,n] ./ nrm
		srcBasis[:,n] = srcBasis[:,n] ./ nrm
		bCoeffs1 .= bCoeffs1 ./ nrm
		# remove projections from source vector
		BLAS.gemv!('N', -1.0 + im*0.0, view(srcBasis, :, 1:(n-1)),
 			view(bCoeffs1, 1:(n-1)), 1.0 + im*0.0, view(srcBasis, :, n))
	end
end
# # perform Gram-Schmidt on target basis, adjusting source basis accordingly
# function gramSchmidtHarm!(trgBasis::Array{T}, srcBasis::Array{T},
# 	bCoeffs1::Vector{T}, bCoeffs2::Vector{T}, opt::Array{T}, n::Integer,
# 	tol::Float64) where T <: Number
# 	# dimension of vector space
# 	dim = size(trgBasis)[1]
# 	# initialize projection norm
# 	prjNrm = 1.0
# 	# initialize projection coefficient memory
# 	bCoeffs1[1:(n-1)] .= 0.0 + im*0.0
# 	# check that basis does not exceed dimension
# 	if n > dim
# 		error("Requested basis size exceeds dimension of vector space.")
# 	end
# 	# norm of proposed vector
# 	nrm = BLAS.nrm2(dim, view(trgBasis,:,n), 1)
# 	# renormalize new vector
# 	trgBasis[:,n] = trgBasis[:,n] ./ nrm
# 	srcBasis[:,n] = srcBasis[:,n] ./ nrm
# 	# guarded orthogonalization
# 	while prjNrm > (tol * 100) && abs(nrm) > tol
# 		### remove projection into existing basis
#  		# calculate projection coefficients
#  		BLAS.gemv!('C', 1.0 + im*0.0, view(trgBasis, :, 1:(n-1)),
#  			view(trgBasis, :, n), 0.0 + im*0.0,
#  			view(bCoeffs2, 1:(n -1)))
#  		# remove projection coefficients
#  		BLAS.gemv!('N', -1.0 + im*0.0, view(trgBasis, :, 1:(n-1)),
#  			view(bCoeffs2, 1:(n -1)), 1.0 + im*0.0,
#  			view(trgBasis, :, n))
#  		# update total projection coefficients
#  		bCoeffs1 .= bCoeffs2 .+ bCoeffs1
#  		# calculate projection norm
#  		prjNrm = BLAS.nrm2(n-1, bCoeffs2, 1)
#  	end
#  	# remaining norm after removing projections
#  	nrm = BLAS.nrm2(dim, view(trgBasis,:,n), 1)
# 	# check that remaining vector is sufficiently large
# 	if abs(nrm) < tol
# 		# switch to random search direction
# 		rand!(view(srcBasis, :, n))
# 		trgBasis[:, n] = opt * srcBasis[:, n]
# 		gramSchmidtHarm!(trgBasis, srcBasis, bCoeffs1, bCoeffs2,
# 			opt, n, tol)
# 	else
# 		# renormalize
# 		trgBasis[:,n] = trgBasis[:,n] ./ nrm
# 		srcBasis[:,n] = srcBasis[:,n] ./ nrm
# 		bCoeffs1 .= bCoeffs1 ./ nrm
# 		# remove projections from source vector
# 		BLAS.gemv!('N', -1.0 + im*0.0, view(srcBasis, :, 1:(n-1)),
#  			view(bCoeffs1, 1:(n-1)), 1.0 + im*0.0, view(srcBasis, :, n))
# 	end
# end
# pseudo-projections for harmonic Ritz vector calculations
@inline function harmVec(dim::Integer, pTrg::Vector{T}, pSrc::Vector{T},
	prjCoeff::Number, sVec::Array{T})::Array{T} where T <: Number
	return sVec .- ((BLAS.dotc(dim, pTrg, 1, sVec, 1) / prjCoeff) .* pSrc)
end
# This is a biconjugate gradient program without a preconditioner.
# m is the maximum number of iterations
function bad_bicgstab_matrix(A, theta, hTrg, hSrc, prjC, b)
	dim = size(A)[1]
    v_m1 = p_m1 = xk_m1 = zeros(ComplexF64,length(b),1)
    # tol = 1e-4a
    # Ax=0 since the initial xk is 0
    r0 = r_m1 = b
    rho_m1 = alpha = omega_m1 = 1
    # for k in 1:length(b)
    # we don't want a perfect solve, should fix this though
    for k in 1 : 16
        rho_k = conj.(transpose(r0))*r_m1
        # beta calculation
        # First term
        first_term = rho_k/rho_m1
        # Second term
        second_term = alpha/omega_m1
        # Calculation
        beta = first_term*second_term
        pk = r_m1 .+ beta.*(p_m1-omega_m1.*v_m1)
        pkPrj = harmVec(dim, hTrg, hSrc, prjC, pk)
        vk = harmVec(dim, hTrg, hSrc, prjC,
        	A * pkPrj .- (theta .* pkPrj))
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
        bPrj = harmVec(dim, hTrg, hSrc, prjC, b)
        t = harmVec(dim, hTrg, hSrc, prjC,
        	A * bPrj .- (theta .* bPrj))
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
### testing
# opt = [8.0 + im*0.0  -3.0 + im*0.0  2.0 + im*0.0;
# 	-1.0 + im*0.0  3.0 + im*0.0  -1.0 + im*0.0;
# 	1.0 + im*0.0  -1.0 + im*0.0  4.0 + im*0.0]

# For RND tests 
sz = 10
opt = Array{ComplexF64}(undef,sz,sz)
rand!(opt)

# For tests
# opt = [2.0 + im*0.0  -2.0 + im*0.0  0.0 + im*0.0;
# 	-1.0 + im*0.0  3.0 + im*0.0  -1.0 + im*0.0;
# 	0.0 + im*0.0  -1.0 + im*0.0  4.0 + im*0.0]
opt[:,:] .= (opt .+ adjoint(opt)) ./ 2
trueEigSys = eigen(opt)
julia_eigvals = trueEigSys.values 
# print("julia_eigvals ", julia_eigvals,"\n")
# Nic
# print("trueEigSys ", trueEigSys, "\n")
# Let's find the smallest positive eigenvalue found using the Julia solver
global julia_min_eigval = 1000
global position = 1
for i in eachindex(julia_eigvals)
	# print("real(julia_eigvals[i]) ", real(julia_eigvals[i]), "\n")
	# print("min_eigval ", min_eigval, "\n")
	if 0 < real(julia_eigvals[i]) < julia_min_eigval
		theta_tilde = real(julia_eigvals[i])
		global position = i 
		global julia_min_eigval = real(theta_tilde)
		# print("julia_min_eigval ", julia_min_eigval, "\n")
		# print("position ", position, "\n")
		# print("Updated position ", i, " times \n")
	end 
	# print("i ", i, "\n")
	# print("min_eigval ", min_eigval, "\n")
end  
# Sean 
# This gives the smalles eigenvalue and not the smallest positive eigenvalue 
# minEigPos = argmin(abs.(trueEigSys.values))
# minEig = trueEigSys.values[minEigPos]
# println("Julia smallest positive eigenvalue is ", minEig,".")
dims = size(opt)
bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
kMat = zeros(ComplexF64, dims[2], dims[2])
loopDim = 2
# Function call for Harmonic Ritz algorithm without restart
# eigval_basic = jacDavRitzHarm_basic(trgBasis, srcBasis, kMat, opt, dims[1],dims[2] , loopDim, 1.0e-4)

# jacDavRitzHarm_restart(trgBasis::Array{ComplexF64}, 
# 	srcBasis::Array{ComplexF64}, kMat::Array{ComplexF64}, 
# 	opt::Array{ComplexF64}, vecDim::Integer, repDim::Integer, 
# 	loopDim::Integer,tol::Float64)::Float64

dims = size(opt)
bCoeffs1 = Vector{ComplexF64}(undef, dims[2])
bCoeffs2 = Vector{ComplexF64}(undef, dims[2])
trgBasis = Array{ComplexF64}(undef, dims[1], dims[2])
srcBasis = Array{ComplexF64}(undef, dims[1], dims[2])
kMat = zeros(ComplexF64, dims[2], dims[2])
innerLoopDim = 4
restartDim = 2
nb_rest_keep_vals = 2
# Function call for Harmonic Ritz algorithm with restart
eigval_restart = jacDavRitzHarm_restart(trgBasis,srcBasis,kMat,opt,dims[1],
	dims[2],restartDim,innerLoopDim,nb_rest_keep_vals,1.0e-4)
# Int(dims[2]/2)
print("No restart - HarmonicRitz smallest positive eigenvalue is ", eigval_basic, "\n")
print("Restart - HarmonicRitz smallest positive eigenvalue is ", eigval_restart, "\n")
println("Julia smallest positive eigenvalue is ", julia_min_eigval,"\n")
