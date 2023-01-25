
# based on the example code from https://tobydriscoll.net/fnc-julia/krylov/gmres.html
# code for the AA case 

# check after the 20 iterations
# with 5 restarts 

using LinearAlgebra
# m is the maximum number of iterations
function davidson(A) # , m=20 # add cellsB for the BA case 
    n = Int(sqrt(length(A)))
    print("n ", n, "\n")				# Dimension of matrix
    tol = 1e-8				# Convergence tolerance
    mmax = n ÷ 2 # n//2

    k = 5 # number of initial guess vectors  
    eig = 2 # number of eigenvalues to solve  

    # t
    t = zeros(Int8, n, k)
    t[1,1] = 1
    t[2,2] = 1
    t[3,3] = 1
    # t = ones(Float64,n,n) # 1* Matrix(I, n, k) # I # np.eye(n,k) # set of k unit vectors as guess  

    print("norm(t) ", norm(t[:,1]), "\n")

    # V 
    V = zeros(Int8,n,n) # np.zeros((n,n)) # array of zeros to hold guess vec  
    
    # Id: identity matrix 
    Id = zeros(Int8, n, n)
    Id[1,1] = 1
    Id[2,2] = 1
    Id[3,3] = 1
    # Id = 1* Matrix(I, n, n) # I = np.eye(n) # identity matrix same dimen as A  
    
    for m in range(k,mmax,k)
        if m <= k
            for j in range(1,k)
                print("t[:,j] ", t[:,j], "\n")
                print("norm(t[:,j]) ", norm(t[:,j]), "\n")
                V[:,j] = t[:,j] /norm(t[:,j])
            end 
            theta_old = 1  
        elseif m > k
            theta_old = theta[:eig]
        end 
        V[:,:m],R = qr(V[:,:m])
        T = dot(transpose(V[:,:m]),dot(A,V[:,:m]))
        THETA,S = eig(T)
        idx = sortperm(THETA) # THETA.argsort()
        theta = THETA[idx]
        s = S[:,idx]
        for j in range(1,k)
            w = dot((A - theta[j]*Id),dot(V[:,:m],s[:,j])) 
            q = w/(theta[j]-A[j,j])
            V[:,(m+j)] = q
        end 
        norm = norm(theta[:eig] - theta_old)
        if norm < tol
            break
        end 
    end 
    return theta[:eig] 
    
end

# Test matrix and vector taken from: 
# https://www.l3harrisgeospatial.com/docs/imsl_sp_gmres.html

A = zeros(Int8, 3, 3)
A[1,1] = 2
A[1,2] = -1
A[1,3] = 0
A[2,1] = -1
A[2,2] = 2
A[2,3] = -1 
A[3,1] = 0
A[3,2] = -1
A[3,3] = 2

print("davidson fct call ", davidson(A), "\n")