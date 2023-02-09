module Random_Matrix
using LinearAlgebra
export indefinite, definite
function indefinite(n) # n is the size of the output matrix
    bool = false
    while bool == false
        n_p = 0 #number of positive eigenvalues
        n_n = 0 #number of negative eigenvalues
        #construct the matrix
        A_0 = rand(n,n)
        Ap = A_0 + adjoint(A_0)
        L = Diagonal(rand(-10:10, n,n))
        global A = Ap + L
        #check eigenvalue to see if is it indefinite
        eig = eigvals(A)
        for i in eig
            if i < 0
                n_n += 1
            elseif i > 0
                n_p += 1
            end
            end
            if n_p == 0 || n_n == 0
                bool = false
            else
                bool = true
            end
        end
    return A
    end

function definite(n)
    R = rand(n,n)
    L = LowerTriangular(R)
    A = adjoint(L)*L
    A = A+ 0.01*I #peak disparity factor
    return A
end
end
#end of random matrix generator module

n = 100
A_0 = Random_Matrix.indefinite(n)
A_1 = Random_Matrix.definite(n)
s_0 = rand(n,1)
s_1 = rand(n,1)