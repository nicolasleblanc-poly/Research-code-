module func 
using Rnd_Matrix 
export f
function f(x)
    n = length(x)
    # Indefinite matrix 
    A_0 = indefinite(n)
    # Definite matrix 
    A_1 = definite(n)
    # Random vectors 
    s_0 = rand(n,1)
    s_1 = rand(n,1)

    t = inv(A_0 + A_1*x)*(s_0+x*s_1)
    f_0 = 2*real(adjoint(t)*s_1) - adjoint(t)*A_1*t
    return f_0[1]
end 
x = ones(Int8, 3, 1)
x[1]=1
x[2]=2
x[3]=3
print("f(x) ", f(x), "\n")
end 

