using LinearAlgebra, LinearAlgebra.BLAS, Distributed, FFTW, Cubature, 
Base.Threads, FastGaussQuadrature, MaxGStructs, MaxGCirc, MaxGBasisIntegrals, 
MaxGOpr, Printf, MaxGParallelUtilities, MaxGCUDA, Random, 
product, bfgs_power_iteration_asym_only, dual_asym_only, gmres

# The following latex files explains the different functions of the program
# https://www.overleaf.com/read/yrdmwzjhqqqs

threads = nthreads()
# Set the number of BLAS threads. The number of Julia threads is set as an 
# environment variable. The total number of threads is Julia threads + BLAS 
# threads. VICu is does not call BLAS libraries during threaded operations, 
# so both thread counts can be set near the available number of cores. 
BLAS.set_num_threads(threads)
# Analogous comments apply to FFTW threads. 
FFTW.set_num_threads(threads)
# Confirm thread counts
blasThreads = BLAS.get_num_threads()
fftwThreads = FFTW.get_num_threads()
println("MaxGTests initialized with ", nthreads(), 
	" Julia threads, $blasThreads BLAS threads, and $fftwThreads FFTW threads.")

# New Green function code start 
# Define test volume, all lengths are defined relative to the wavelength. 
# Number of cells in the volume. 
cellsA = [4, 4, 4 ]
cellsB = [1, 1, 1]
# Edge lengths of a cell relative to the wavelength. 
scaleA = (0.1, 0.1, 0.1)
scaleB = (0.2, 0.2, 0.2)
# Center position of the volume. 
coordA = (0.0, 0.0, 0.0)
coordB = (0.0, 0.0, 1.0)
### Prepare Green functions
println("Green function construction started.")
# Create domains
assemblyInfo = MaxGAssemblyOpts()
aDom = MaxGDom(cellsA, scaleA, coordA) 
bDom = MaxGDom(cellsB, scaleB, coordB)
## Prepare Green function operators. 
# First number is adjoint mode, set to zero for standard operation. 
gMemSlfN = MaxGOprGenSlf(0, assemblyInfo, aDom)
gMemSlfA = MaxGOprGenSlf(1, assemblyInfo, aDom)
gMemExtN = MaxGOprGenExt(0, assemblyInfo, aDom, bDom)
# # Operator shorthands
# greenActAA! = () -> grnOpr!(gMemSlfN) 
# greenAdjActAA! = () -> grnOpr!(gMemSlfA)
# greenActAB! = () -> grnOpr!(gMemExtN)
println("Green function construction completed.")
# New Green function code stop 

# New Green function test code 
# Start 
# testvect = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)
# rand!(testvect)

# print("testvect ",test(gMemSlfN, testvect), "\n")

# grnOpr!(gMemSlfN)

# copyto!(gMemSlfN.srcVec, testvect)

# # print("gMemSlfN.srcVec ",gMemSlfN.srcVec, "\n")

# # greenActAA!()

# print("gMemSlfN.trgVec ",gMemSlfN.trgVec, "\n")
# End 


# Source current memory
currSrcAB = Array{ComplexF64}(undef, cellsB[1], cellsB[2], cellsB[3], 3)

# Calculate ei electric field vector 
# elecIn = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3) 
# elecIn_vect = Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3, 1)
# currSrcAB[1,1,1,3] = 10.0 + 0.0im

# Good ei code start 
ei = Gv_AB(gMemExtN, cellsA, cellsB, currSrcAB) # This is a vector, so it is already reshaped
print("ei ", ei, "\n")
# Good ei code end 

# copy!(elecIn,GAB_curr);
# elecIn_reshaped = reshape(elecIn, (cellsA[1]*cellsA[2]*cellsA[3]*3,1))
# print(reshape(elecIn, (cellsA[1]*cellsA[2]*cellsA[3]*3)))
# print(size(reshape(elecIn, (cellsA[1]*cellsA[2]*cellsA[3]*3,1))))
# print(ei)

# End 

# Let's define some values used throughout the program.
# chi coefficient
chi_coeff = 3.0 + 0.01im
# inverse chi coefficient
chi_inv_coeff = 1/chi_coeff 
chi_inv_coeff_dag = conj(chi_inv_coeff)
# define the projection operators
P = I # this is the real version of the identity matrix since we are considering 
# the symmetric and ansymmetric parts of some later calculations. 
# If we were only considering the symmetric parts of some latter calculations,
# we would need to use the imaginary version of the identity matrix. 
# Pdag = conj.(transpose(P)) 
# we could do the code above for a P other than the identity matrix
Pdag = P
# let's get the initial b vector (aka using the initial Lagrange multipliers). Done for test purposes
# b = bv(ei, l,P)

# The bunch of code below was for when I thought it was better to define a set of 
# initial multiplier values and then generate P's. When using the eventual software,
# the user doesn't care about Lagrange multipliers. They only care about constraints
# which are defined by the elements of the diagonal of the P's. 
# Start old code 
# l = [0.75] # Initial Lagrange multipliers associated to the asymmetric constraints
# l2 = [0.25] # Initial Lagrange multipliers associated to the symmetric constraints
# l_sum = length(l)+length(l2)
# print("Sum of lengths ", l_sum, "\n")

# Code to the generation of P matrices that are turned into vectors since the P 
# matrices are 0 except on the main diagonal 

# M = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3) 
# for i=0:l_sum-1
# 	M[i*cellsA[1]/l_sum+1:(i+1)*cellsA[1]/l_sum,i*cellsA[2]/l_sum+1:(i+1)*cellsA[2]/l_sum,i*cellsA[3]/l_sum+1:(1+i)*cellsA[3]/l_sum,:] = 1
# end 
# Problems/questions:
# 1. I added a *3 to the size of M because we need it and you didn't say it before, so I just wanted to make sure.
# I guess we reset M at each iteration, so whenever we generate a new P?
# 2. How to use linear indexing to only get the diagonal of the M matrix for a given subsection?
# 3. I guess we need to use an element-wise multiplication for the P*(chi^{-1 \dag}-G_0^{\dag}) calculation?
# 4. Davidson iteration program needs an explicit multiplier solver, which I find weird 
# End of old code 

# Start of new code (Saturday, January 21st 2023 at 19:37)
# Let's start by generating the P's. The most simple and logical situation 
# for splitting the cubic domain (so in 3D) is to cut the domain in x, in y
# and in z in half. Let's assume that the amount of cells is always even
# to avoid having decimal terms when we divide by 2. We then get 8 little/baby cubes. 
# The indexes will be [1:half] or [half:end]. For all of the baby cubes, the idea is 
# the same. We first need to create an array that we shall call M. M has the same dimensions of G 
# (the Green function), so (N_x,N_y,N_z,3) where N_i is the amount of cells in
# the i direction. To clarify, Nx=cellsA[1], Ny=cellsA[2], Nz=cellsA[3].
# Next, we need to select a baby cube using linear indexing and 
# add 1's everywhere in this part of the total array (the part associated with the 
# baby cube). We can then use diag(M) to get the diagonal of the diagonal of MaxGAssemblyOpts
# as a vector that we can then use in the rest of the program. Yeah! :)
# Not sure how to do the multiple lines of code below in a loop.
# The loop would be needed to simplify everything and to add 
# the depth parameter that would allow to add another division 
# by 2 in the x, y and z direction, which would be a depth of 2. This would allow to have 64
# constraints instead of 8. We could also do this for a depth of 3, where we could have 
# 512 constraints, which is kinda getting extreme but nonetheless pertinent. 
# First baby cube: [1:cellsA[1]/2, 1:cellsA[2]/2, 1:cellsA[3]/2]
M = zeros(cellsA[1],cellsA[2],cellsA[3],3)
M[1:Int(cellsA[1]/2), 1:Int(cellsA[2]/2), 1:Int(cellsA[3]/2),:] .= 1.0
print("1:Int(cellsA[1]/2) ", 1:Int(cellsA[1]/2), "\n")
print("1:Int(cellsA[2]/2) ", 1:Int(cellsA[2]/2), "\n")
print("1:Int(cellsA[3]/2) ", 1:Int(cellsA[3]/2), "\n")
P1 = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1))
# N = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1)) # diag(M)
# print("N ", N, "\n")
# N_diag = Diagonal(N)
# P1 = zeros(cellsA[1]*cellsA[2]*cellsA[3]*3, 1) # Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3, 1)
# P1[1:length(N_diag)] = N_diag[1:end] 
# # Diagonal only stores a value from N if its non-zero. 
# # test_vec = ones(cellsA[1]*cellsA[2]*cellsA[3]*3,1)
print("P1 ", P1, "\n")
# # print("size(P1) ", size(P1), "\n")
# # print("test_vec ", test_vec, "\n")
# print("size(test_vec) ", size(test_vec), "\n")
# print("P1*test_vec ", P1.*test_vec, "\n")

# Second baby cube: [1:cellsA[1]/2, 1:cellsA[2]/2, cellsA[3]/2+1:end]
M = zeros(cellsA[1],cellsA[2],cellsA[3],3)
M[1:Int(cellsA[1]/2), 1:Int(cellsA[2]/2), Int(cellsA[3]/2)+1:end,:] .= 1.0
# print("1:Int(cellsA[1]/2) ", 1:Int(cellsA[1]/2), "\n")
# print("1:Int(cellsA[2]/2) ", 1:Int(cellsA[2]/2), "\n")
# print("1:Int(cellsA[3]/2) ", Int(cellsA[3]/2)+1:, "\n")
P2 = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1))
# N = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1)) # diag(M)
# print("N ", N, "\n")
# N_diag = Diagonal(N)
# P2 = zeros(cellsA[1]*cellsA[2]*cellsA[3]*3, 1) # Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3, 1)
# P2[1:length(N_diag)] = N_diag[1:end] 
print("P2 ", P2, "\n")

# Third baby cube: [1:cellsA[1]/2, cellsA[2]/2+1:end, 1:cellsA[3]/2]
M = zeros(cellsA[1],cellsA[2],cellsA[3],3)
M[1:Int(cellsA[1]/2), Int(cellsA[2]/2)+1:end, 1:Int(cellsA[3]/2),:] .= 1.0
P3 = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1))
# N = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1)) # diag(M)print("N ", N, "\n")
# print("N ", N, "\n")
# N_diag = Diagonal(N)
# P3 = zeros(cellsA[1]*cellsA[2]*cellsA[3]*3, 1) # Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3, 1)
# P3[1:length(N_diag)] = N_diag[1:end] 
print("P3 ", P3, "\n")

# Fourth baby cube: [1:cellsA[1]/2, cellsA[2]/2+1:end, cellsA[3]/2+1:end]
M = zeros(cellsA[1],cellsA[2],cellsA[3],3)
M[1:Int(cellsA[1]/2), Int(cellsA[2]/2)+1:end, Int(cellsA[3]/2)+1:end,:] .= 1.0
P4 = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1))
# N = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1)) # diag(M)
# print("N ", N, "\n")
# N_diag = Diagonal(N)
# P4 = zeros(cellsA[1]*cellsA[2]*cellsA[3]*3, 1) # Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3, 1)
# P4[1:length(N_diag)] = N_diag[1:end] 
print("P4 ", P4, "\n")

# Fifth baby cube: [cellsA[1]/2+1:end, 1:cellsA[2]/2, 1:cellsA[3]/2]
M = zeros(cellsA[1],cellsA[2],cellsA[3],3)
M[Int(cellsA[1]/2)+1:end, 1:Int(cellsA[2]/2), 1:Int(cellsA[3]/2),:] .= 1.0
P5 = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1))
# N = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1)) # diag(M)
# print("N ", N, "\n")
# N_diag = Diagonal(N)
# P5 = zeros(cellsA[1]*cellsA[2]*cellsA[3]*3, 1) # Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3, 1)
# P5[1:length(N_diag)] = N_diag[1:end] 
print("P5 ", P5, "\n")

# Sixth baby cube: [cellsA[1]/2+1:end, cellsA[2]/2+1:end, 1:cellsA[3]/2]
M = zeros(cellsA[1],cellsA[2],cellsA[3],3)
M[Int(cellsA[1]/2)+1:end, Int(cellsA[2]/2)+1:end, 1:Int(cellsA[3]/2),:] .= 1.0
P6 = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1))
# N = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1)) # diag(M)
# print("N ", N, "\n")
# N_diag = Diagonal(N)
# P6 = zeros(cellsA[1]*cellsA[2]*cellsA[3]*3, 1) # Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3, 1)
# P6[1:length(N_diag)] = N_diag[1:end] 
print("P6 ", P6, "\n")

# Seventh baby cube: [cellsA[1]/2+1:end, 1:cellsA[2]/2, cellsA[3]/2+1:end]
M = zeros(cellsA[1],cellsA[2],cellsA[3],3)
M[Int(cellsA[1]/2)+1:end, 1:Int(cellsA[2]/2), Int(cellsA[3]/2)+1:end,:] .= 1.0
P7 = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1))
# N = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1)) # diag(M)
# print("N ", N, "\n")
# N_diag = Diagonal(N)
# P7 = zeros(cellsA[1]*cellsA[2]*cellsA[3]*3, 1) # Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3, 1)
# P7[1:length(N_diag)] = N_diag[1:end]
print("P7 ", P7, "\n") 

# Eigth baby cube: [cellsA[1]/2+1:end, cellsA[2]/2+1:end, cellsA[3]/2+1:end]
M = zeros(cellsA[1],cellsA[2],cellsA[3],3)
M[Int(cellsA[1]/2)+1:end, Int(cellsA[2]/2)+1:end, Int(cellsA[3]/2)+1:end,:] .= 1.0
P8 = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1))
# N = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1)) # diag(M))
# print("N ", N, "\n")
# N_diag = Diagonal(N)
# print("N_diag ", N_diag, "\n")
# P8 = zeros(cellsA[1]*cellsA[2]*cellsA[3]*3, 1) # Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3, 1)
# for i in eachindex(N_diag)
# 	if N_diag[i] != 0.0
# 		P8[i] = 1.0
# 	end 
# end 
# # P8[1:length(N_diag)] = N_diag[1:end]
print("P8 ", P8, "\n")


# For simplicity, I'm going to say that the first constraints are asymmetric and the ones 
# after are symmetric 
P = [P1,P2,P3,P4,P5,P6,P7,P8]
# Is P always real?

number_asym_constraints = 4
number_sym_constraints = 4
l = Array{ComplexF64}(undef, number_asym_constraints,1) # L mults related to asym constraints 
l2 = Array{ComplexF64}(undef, number_sym_constraints, 1) # L mults related to sym constraints
# Let's attribute random starting Lagrange multipliers that are between 0 and 3 (kinda 
# arbitrary but we know the L mults are generally small)
rand!(l,(0.01:3))
rand!(l2,(0.01:3))

# End of new code for generating P's and then Lagrange multipliers 

# This is the code for the main function call using bfgs with the power iteration
# method to solve for the Lagrange multiplier and gmres to solve for |T>.
# # Start 
bfgs = BFGS_fakeS_with_restart_pi(gMemSlfN,gMemSlfA,l,l2,dual,P,chi_inv_coeff,ei,
cellsA,validityfunc,power_iteration_second_evaluation)
# the BFGS_fakeS_with_restart_pi function can be found in the bfgs_power_iteration_asym_only file
dof = bfgs[1]
grad = bfgs[2]
dualval = bfgs[3]
objval = bfgs[4]
print("dof ", dof, "\n")
print("grad ", grad, "\n")
print("dualval ", dualval, "\n")
print("objval ", objval, "\n")
# End 


# ITEM call code 
# item = ITEM(gMemSlfN, gMemSlfA,l,dual,P,chi_inv_coeff,ei,cellsA,validityfunc) 
# print("dof ", item[1], "\n")
# print("grad ", item[2], "\n")
# print("dualval ", item[3], "\n")
# print("objval ", item[4], "\n")


# testvect = Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3,1)
# rand!(testvect)
# print("Gv_AA(gMemSlfN, cellsA, vec) ", Gv_AA(gMemSlfN, cellsA, testvect), "\n")
# print("l[1]*asym_vect(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,testvect)", l[1]*asym_vect(gMemSlfN,gMemSlfA,cellsA,chi_inv_coeff,P,testvect)
# , "\n")