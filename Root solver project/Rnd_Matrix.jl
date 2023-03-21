module Rnd_Matrix
using LinearAlgebra
export indefinite, definite
function indefinite(n) # n is the size of the output matrix
    # print("1 \n")
    bool = false
    while bool == false
        n_p = 0 # Number of positive eigenvalues
        n_n = 0 # Number of negative eigenvalues
        # Construct the matrix
        A_0 = rand(n,n)
        Ap = A_0 + adjoint(A_0)
        L = Diagonal(rand(-10:10, n,n))
        global A = Ap + L
        # print("A ", A, "\n")
        # Check eigenvalue to see if is it indefinite
        eig = eigvals(A)
        for i in eig
            if i < 0
                n_n += 1
            elseif i > 0
                n_p += 1
            end
        end
        # print("n_n ", n_n, "\n")
        # print("n_p ", n_p, "\n")
        if n_p == 0 || n_n == 0
            bool = false
        else
            bool = true
        end
        # print("2 \n")
    end
    return A
end

function definite(n)
    R = rand(n,n)
    L = LowerTriangular(R)
    A = adjoint(L)*L
    A = A+ 0.01*I # Peak disparity factor
    return A
end

#end of random matrix generator module

# n = 100
# A_0 = Random_Matrix.indefinite(n)
# A_1 = Random_Matrix.definite(n)
# s_0 = rand(n,1)
# s_1 = rand(n,1)

n = 15
A_0 = indefinite(n)
A_1 = definite(n)
s_0 = rand(n,1)
s_1 = rand(n,1)
println(A_0)
println(A_1)
println(s_0)
println(s_1)

end