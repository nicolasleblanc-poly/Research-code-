using LinearAlgebra, LinearAlgebra.BLAS, Distributed, FFTW, Cubature, 
Base.Threads, FastGaussQuadrature, MaxGStructs, MaxGCirc, MaxGBasisIntegrals, 
MaxGOpr, Printf, MaxGParallelUtilities, MaxGCUDA, Random
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