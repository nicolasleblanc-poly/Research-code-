# This istance modifies the A_1 matrix to add bigger diagonal entries

# Start of random matrix generator module
module Random_Matrix
using LinearAlgebra
using Plots
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


n = 15
A_0 = Random_Matrix.indefinite(n)
A_1 = Random_Matrix.definite(n)
s_0 = rand(n,1)
s_1 = rand(n,1)

# println(A_0)
# println(A_1)
# println(s_0)
# println(s_1)

function f(x)
    t = inv(A_0 + x*A_1)*(s_0+x*s_1)
    f_0 =  2*real(adjoint(t)*s_1) - adjoint(t)*A_1*t 
    return f_0[1]
end



#plots the piecewise function
xmax = 1000
xmin = 0
px = range(xmax, xmin, 5000)
py = map(f, px)
ymax = 100
ymin = -500
lbound = 0
global plt = plot(px,py, ylim=(ymin,ymax), legend = false)
display(plt)


end
#end of random matrix generator module
