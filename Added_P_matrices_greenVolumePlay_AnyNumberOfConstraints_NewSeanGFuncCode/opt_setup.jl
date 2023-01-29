module opt_setup 
export Ps, L_mults
using LinearAlgebra, LinearAlgebra.BLAS, Distributed, FFTW, Cubature, 
Base.Threads, FastGaussQuadrature, MaxGStructs, MaxGCirc, MaxGBasisIntegrals, 
MaxGOpr, Printf, MaxGParallelUtilities, MaxGCUDA, Random, 
product, bfgs_power_iteration_asym_only, dual_asym_only, gmres

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

# 0th P. This P is the complex identity matrix and it is considered because it assures 
# duality when using one asym constraint. 


function Ps(cellsA, nb_complex_P, nb_real_P)
    M = zeros(cellsA[1],cellsA[2],cellsA[3],3)
    M[1:Int(cellsA[1]/2), 1:Int(cellsA[2]/2), 1:Int(cellsA[3]/2),:] .= 1.0im
    print("1:Int(cellsA[1]/2) ", 1:Int(cellsA[1]/2), "\n")
    print("1:Int(cellsA[2]/2) ", 1:Int(cellsA[2]/2), "\n")
    print("1:Int(cellsA[3]/2) ", 1:Int(cellsA[3]/2), "\n")
    N = reshape(M, cellsA[1]*cellsA[2]*cellsA[3]*3)
    P1 = Diagonal(N)
    # P1 = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1))
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
    M[1:Int(cellsA[1]/2), 1:Int(cellsA[2]/2), Int(cellsA[3]/2)+1:end,:] .= 1.0im
    # print("1:Int(cellsA[1]/2) ", 1:Int(cellsA[1]/2), "\n")
    # print("1:Int(cellsA[2]/2) ", 1:Int(cellsA[2]/2), "\n")
    # print("1:Int(cellsA[3]/2) ", Int(cellsA[3]/2)+1:, "\n")
    N = reshape(M, cellsA[1]*cellsA[2]*cellsA[3]*3)
    P2 = Diagonal(N)
    # P2 = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1))
    # N = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1)) # diag(M)
    # print("N ", N, "\n")
    # N_diag = Diagonal(N)
    # P2 = zeros(cellsA[1]*cellsA[2]*cellsA[3]*3, 1) # Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3, 1)
    # P2[1:length(N_diag)] = N_diag[1:end] 
    print("P2 ", P2, "\n")

    # Third baby cube: [1:cellsA[1]/2, cellsA[2]/2+1:end, 1:cellsA[3]/2]
    M = zeros(cellsA[1],cellsA[2],cellsA[3],3)
    M[1:Int(cellsA[1]/2), Int(cellsA[2]/2)+1:end, 1:Int(cellsA[3]/2),:] .= 1.0im
    N = reshape(M, cellsA[1]*cellsA[2]*cellsA[3]*3)
    P3 = Diagonal(N)
    # P3 = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1))
    # N = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1)) # diag(M)print("N ", N, "\n")
    # print("N ", N, "\n")
    # N_diag = Diagonal(N)
    # P3 = zeros(cellsA[1]*cellsA[2]*cellsA[3]*3, 1) # Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3, 1)
    # P3[1:length(N_diag)] = N_diag[1:end] 
    print("P3 ", P3, "\n")

    # Fourth baby cube: [1:cellsA[1]/2, cellsA[2]/2+1:end, cellsA[3]/2+1:end]
    M = zeros(cellsA[1],cellsA[2],cellsA[3],3)
    M[1:Int(cellsA[1]/2), Int(cellsA[2]/2)+1:end, Int(cellsA[3]/2)+1:end,:] .= 1.0im
    N = reshape(M, cellsA[1]*cellsA[2]*cellsA[3]*3)
    P4 = Diagonal(N)
    # P4 = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1))
    # N = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1)) # diag(M)
    # print("N ", N, "\n")
    # N_diag = Diagonal(N)
    # P4 = zeros(cellsA[1]*cellsA[2]*cellsA[3]*3, 1) # Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3, 1)
    # P4[1:length(N_diag)] = N_diag[1:end] 
    print("P4 ", P4, "\n")

    # Fifth baby cube: [cellsA[1]/2+1:end, 1:cellsA[2]/2, 1:cellsA[3]/2]
    M = zeros(cellsA[1],cellsA[2],cellsA[3],3)
    M[Int(cellsA[1]/2)+1:end, 1:Int(cellsA[2]/2), 1:Int(cellsA[3]/2),:] .= 1.0
    N = reshape(M, cellsA[1]*cellsA[2]*cellsA[3]*3)
    P5 = Diagonal(N)
    # P5 = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1))
    # N = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1)) # diag(M)
    # print("N ", N, "\n")
    # N_diag = Diagonal(N)
    # P5 = zeros(cellsA[1]*cellsA[2]*cellsA[3]*3, 1) # Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3, 1)
    # P5[1:length(N_diag)] = N_diag[1:end] 
    print("P5 ", P5, "\n")

    # Sixth baby cube: [cellsA[1]/2+1:end, cellsA[2]/2+1:end, 1:cellsA[3]/2]
    M = zeros(cellsA[1],cellsA[2],cellsA[3],3)
    M[Int(cellsA[1]/2)+1:end, Int(cellsA[2]/2)+1:end, 1:Int(cellsA[3]/2),:] .= 1.0
    N = reshape(M, cellsA[1]*cellsA[2]*cellsA[3]*3)
    P6 = Diagonal(N)
    # P6 = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1))
    # N = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1)) # diag(M)
    # print("N ", N, "\n")
    # N_diag = Diagonal(N)
    # P6 = zeros(cellsA[1]*cellsA[2]*cellsA[3]*3, 1) # Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3, 1)
    # P6[1:length(N_diag)] = N_diag[1:end] 
    print("P6 ", P6, "\n")

    # Seventh baby cube: [cellsA[1]/2+1:end, 1:cellsA[2]/2, cellsA[3]/2+1:end]
    M = zeros(cellsA[1],cellsA[2],cellsA[3],3)
    M[Int(cellsA[1]/2)+1:end, 1:Int(cellsA[2]/2), Int(cellsA[3]/2)+1:end,:] .= 1.0
    N = reshape(M, cellsA[1]*cellsA[2]*cellsA[3]*3)
    P7 = Diagonal(N)
    # P7 = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1))
    # N = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3,1)) # diag(M)
    # print("N ", N, "\n")
    # N_diag = Diagonal(N)
    # P7 = zeros(cellsA[1]*cellsA[2]*cellsA[3]*3, 1) # Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3, 1)
    # P7[1:length(N_diag)] = N_diag[1:end]
    print("P7 ", P7, "\n") 

    # Eigth baby cube: [cellsA[1]/2+1:end, cellsA[2]/2+1:end, cellsA[3]/2+1:end]
    M = zeros(cellsA[1],cellsA[2],cellsA[3],3)
    M[Int(cellsA[1]/2)+1:end, Int(cellsA[2]/2)+1:end, Int(cellsA[3]/2)+1:end,:] .= 1.0
    N = reshape(M, cellsA[1]*cellsA[2]*cellsA[3]*3)
    P8 = Diagonal(N)
    # N = reshape(M, (cellsA[1]*cellsA[2]*cellsA[3]*3)) # diag(M))
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
    # Is P always real? -> No 


    multipliers = L_mults(nb_complex_P,nb_real_P)

    return P, multipliers 
end 

function L_mults(nb_complex_P,nb_real_P)
    # number_asym_constraints = 4
    # number_sym_constraints = 4
    l = Array{ComplexF64}(undef, nb_complex_P,1) # L mults related to asym constraints 
    l2 = Array{ComplexF64}(undef, nb_real_P, 1) # L mults related to sym constraints
    # Let's attribute random starting Lagrange multipliers that are between 0 and 3 (kinda 
    # arbitrary but we know the L mults are generally small)
    rand!(l,(0.01:3))
    rand!(l2,(0.01:3))
    return l, l2
end 
end 
