using LinearAlgebra, Plots
function indefinite(n) # n is the size of the output matrix
    # print("1 \n")
    bool = false
    while bool == false
        n_p = 0 #number of positive eigenvalues
        n_n = 0 #number of negative eigenvalues
        #construct the matrix
        A_0 = rand(n,n)
        Ap = A_0 + adjoint(A_0)
        L = Diagonal(rand(-10:10, n,n))
        global A = Ap + L
        # print("A ", A, "\n")
        #check eigenvalue to see if is it indefinite
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
    A = A+ 0.01*I #peak disparity factor
    return A
end


function f(x)
    # print("In func \n")
    n=100
    # n = length(x)
    
    # Indefinite matrix 
    A_0 = indefinite(n)
    # print("A0 done \n")
    # Definite matrix 
    A_1 = definite(n)
    # print("A1 done \n")
    # Random vectors 
    s_0 = rand(n,1)
    s_1 = rand(n,1)

    # print("A_0 ",A_0, "\n")
    # print("A_1 ",A_1, "\n")
    # print("s_0 ",s_0, "\n")
    # print("x ",x, "\n")
    # print(" ", , "\n")
    # print(" ", , "\n")
    # print(" ", , "\n")

    t = inv(A_0 + A_1*x)*(s_0+x*s_1)
    f_0 = 2*real(adjoint(t)*s_1) - adjoint(t)*A_1*t
    return f_0[1]
end 

# function 

# end 
print("f(1000) ", f(5000),"\n")

# Create a julia equivalent of a linspace 
# x=2
x = range(0,1000,100) # start,stop,number of points  
# fx = f(x)
graph = plot(x,f)
yaxis!(yvals=[-500:100])
# savefig(graph) # save the most recent fig as filename_string (such as "output.png")
display(graph)