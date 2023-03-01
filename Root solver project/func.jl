module func 
using Rnd_Matrix, Plots
export f
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
print("f(1000) ", f(1000),"\n")

# Create a julia equivalent of a linspace 
# x=2
x = range(0,1000,100) # start,stop,number of points  
# fx = f(x)
# graph = plot(x,f)
# savefig(graph) # save the most recent fig as filename_string (such as "output.png")
# display(graph)
# print("f(x) ", f(x), "\n")
end 

