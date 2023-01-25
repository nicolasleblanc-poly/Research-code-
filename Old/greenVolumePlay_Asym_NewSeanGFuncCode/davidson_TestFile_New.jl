using LinearAlgebra, Random, Arpack

function modified_gram_schmidt(V, t)
    for j = 1:size(V)[2]
        vj = V[1:end,j]
        qj = vj/norm(vj)
        for k = j+1:size(V)[2]
            vk = vk - (conj.(transpose(vk))*qj)*qj
        end
    end
    return t = t-(conj.(transpose(t))*vk)*vk
end 

function davidson_it(A)
    # m = 20 # Amount of iterations of the inner loop

    tol = 1e-15 # Tolerance for which the program will converge 

    rows = size(A)[1]
    cols = size(A)[2]

    v = Array{Float64}(undef, cols,1)
    rand!(v)

    vk = v/norm(v) # Vector 
    wk = A*v # Vector 
    hk = (conj.(transpose(vk))*wk)[1] # Number

    Vk = Array{Float64}(undef, rows, cols)
    # The first index is to select a row
    # The second index is to select a column
    Vk[1:end,1] = vk

    Wk = Array{Float64}(undef, rows, cols)
    Wk[1:end,1] = wk
    
    Hk = Array{Float64}(undef, rows, cols)
    # print("hk ", hk[1], "\n")
    Hk[1,1] = hk

    u = vk # Vector 
    theta = hk # Number 
    r = wk - theta*u # Vector

    u_hat = Array{Float64}(undef, cols, 1)

    for index = 1:10000000
        for i = 2:cols
            t = ((I-u*conj.(transpose(u)))*(A-theta*I)*(I-u*conj.(transpose(u))))\(-r)
    
            # Orthogonalize t against V_k using MGS 
            # Expand V_k with this vector to V_{k+1}
            # Temporary value for the new v_k (aka v_{k+1})
            vk = Array{Float64}(undef, cols,1)
            rand!(v)
    
            wk = A*vk 
            # Expand W_k with this vector to W_{k+1}
            Wk[1:end,i] = wk

            # print("conj.(transpose(A)) ", conj.(transpose(A)), "\n")
            if A != conj.(transpose(A))
                print("(conj.(transpose(Vk))*wk)[1] ", (conj.(transpose(Vk))*wk)[1], "\n")
                Hk[1:end,i] = (conj.(transpose(Vk))*wk)[1]
                Hk[end, 1:end] = (conj.(transpose(vk))*Wk)[1]
    
                # Compute the largest eigenpar (theta,s) of H_{k+1} 
                # with the norm(s) = 1
                theta, eigenvector = eigs(Hk, which=:LM)
                s = eigenvector/norm(eigenvector) # normalized_eigenvector
    
                u = Vk*s # Compute the Ritz vector u 
                global u_hat = A*u # Should also have: A*u = W_k*s
    
                r = u_hat - theta*u # Residual vector 
            end 
        end
        if norm(r) <= tol
            print("Exited the loop using break")
            break 
        end 
        Vk = Array{Float64}(undef, rows, cols)
        Vk[1:end,1] = vk
    
        Wk = Array{Float64}(undef, rows, cols)
        Wk[1:end,1] = u_hat
        
        Hk = Array{Float64}(undef, rows, cols)
        Hk[1,1] = theta[1]
    end
    return theta
end 

A = Array{Float64}(undef, 3, 3)
# A = zeros(Int8, 3, 3)
A[1,1] = 2
A[1,2] = -1
A[1,3] = 0
A[2,1] = -1
A[2,2] = 2
A[2,3] = -1 
A[3,1] = 0
A[3,2] = -1
A[3,3] = 2
print("A ", A, "\n")
# print(A[1,:])
# print("size(A) ", size(A)[2], "\n")

# print(eigen(A).values)

print(davidson_it(A), "\n")