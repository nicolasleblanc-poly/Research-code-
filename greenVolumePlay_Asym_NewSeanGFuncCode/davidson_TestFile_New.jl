module davidson_TestFile_New
using LinearAlgebra, Random, Arpack

# function modified_gram_schmidt(V, t, i)
#     # vk = 0
#     print("V ", V, "\n")
#     for j = 1:i-1 # size(V)[2]
#         vj = V[1:end,j]
#         print("vj ", vj, "\n")
#         global qj = vj/norm(vj)
#         print("qj ", qj, "\n")
#         print("(conj.(transpose(qj))*vk) ", (conj.(transpose(qj))*vj)[1], "\n")
#         # for k = 1:j-1
#         for k = j+1:i
#         # for k = j+1:size(V)[2]
#             print("(conj.(transpose(qj))*vk)*qj ", (conj.(transpose(qj))*vj)[1]*qj, "\n")
#             vj = vj - ((conj.(transpose(qj))*vj)[1])*qj
#             # vk = vk - (conj.(transpose(vk))*qj)*qj
#         end
#         # print("dot product ", dot(vj,qj),"\n")
#     end
#     t = t-((conj.(transpose(qj))*t)[1])*qj
#     print("dot product ", dot(t,qj),"\n")

#     # print("product ", conj.transpose(vk), t,"\n")
#     return t = t-((conj.(transpose(qj))*t)[1])*qj
#     # return t = t-(conj.(transpose(t))*vk)*vk
# end 

function modified_gram_schmidt(A, t, index) # i
    # orthogonalises the columns of the input matrix
    print("size(A) ", size(A), "\n")
    print("index ", index, "\n")
    print("A ", A, "\n")
    print("A val ", A[:,1], "\n")
    matrix = Array{Float64}(undef, size(A)[1],index+size(t)[2]) # size(A)[1]
    print("matrix ", matrix, "\n")
    print("matrix[:,1:size(A)[1]] ", matrix[:,1:index], "\n")
    # i is the amount of cols (which are vectors) of A that we are considering here 
    matrix[:,1:index] = A #  matrix[:,1:size(A)[2]] = A
    matrix[:,index+1:end] = t

    print("size(matrix) ", size(matrix),"\n")
    print("matrix ", matrix, "\n")
    num_vectors = size(matrix)[2]
    orth_matrix = copy(matrix)
    for vec_idx = 1:num_vectors
        orth_matrix[:, vec_idx] = orth_matrix[:, vec_idx]/norm(orth_matrix[:, vec_idx])
        for span_base_idx = (vec_idx+1):num_vectors
            # perform block step
            orth_matrix[:, span_base_idx] -= dot(orth_matrix[:, span_base_idx], orth_matrix[:, vec_idx])*orth_matrix[:, vec_idx]
        end
    end
    print("orth_matrix ", orth_matrix, "\n")
    return orth_matrix[:,end:end]
    # return orth_matrix
end

function davidson_it(A)
    # m = 20 # Amount of iterations of the inner loop

    tol = 1e-15 # Tolerance for which the program will converge 

    rows = size(A)[1]
    cols = size(A)[2]

    v = zeros(Float64, rows, 1)
    # v[1,1] = 1
    # v[2,1] = 2
    # v[3,1] = 3
    # v = Array{Float64}(undef, rows,1)
    rand!(v)
    print("v ", v, "\n")

    vk = v/norm(v) # Vector 
    wk = A*v # Vector 
    hk = (conj.(transpose(vk))*wk)[1] # Number
    print("hk ", hk, "\n")

    # Vk = Array{Float64}(undef, rows, cols)
    Vk = zeros(Float64, rows, cols)
    # The first index is to select a row
    # The second index is to select a column
    Vk[:,1] = vk

    # Wk = Array{Float64}(undef, rows, cols)
    Wk = zeros(Float64, rows, cols)
    Wk[:,1] = wk
    
    Hk = zeros(Float64, rows, cols)
    # Hk = Array{Float64}(undef, rows, cols)
    # print("hk ", hk[1], "\n")
    Hk[1,1] = hk

    u = vk # Vector 
    theta = hk # Number 
    r = wk - theta*u # Vector
    # t = Array{Float64}(undef, rows, 1)
    t = zeros(Float64, rows, 1)

    # u_hat = Array{Float64}(undef, rows, 1)
    u_hat = zeros(Float64, rows, 1)

    for index = 1:1000
        for i = 2:cols # Old version 
        # for i = 1:cols # New version 
            while det(((I-u*conj.(transpose(u)))*(A-theta[1]*I)*(I-u*conj.(transpose(u))))) == 0.0+0.0im
                rand!(v)
            end
            print("det ", det(((I-u*conj.(transpose(u)))*(A-theta[1]*I)*(I-u*conj.(transpose(u))))),"\n")
            print("((I-u*conj.(transpose(u)))*(A-theta[1]*I)*(I-u*conj.(transpose(u)))) ", ((I-u*conj.(transpose(u)))*(A-theta[1]*I)*(I-u*conj.(transpose(u)))), "\n")
            t = ((I-u*conj.(transpose(u)))*(A-theta[1]*I)*(I-u*conj.(transpose(u))))\(-r)
    
            # Orthogonalize t against V_k using MGS 
            # Temporary value for the new v_k (aka v_{k+1})
            # vk = Array{Float64}(undef, rows,1)
            # vk = zeros(Float64, rows, 1)
            # rand!(v)
            index = i-1
            print("Vk[:,index] ", Vk[:,1:index], "\n")
            vk = modified_gram_schmidt(Vk[:,1:index], t, index) # modified_gram_schmidt(Vk, t, i)
            print("vk ", vk, "\n")
            # Expand V_k with this vector to V_{k+1}
            Vk[:,i] = vk
            # V_{k+1} = vk 

            wk = A*vk 
            # Expand W_k with this vector to W_{k+1}
            Wk[:,i] = wk

            # print("conj.(transpose(A)) ", conj.(transpose(A)), "\n")
            # print("A ", A, "\n")
            # print("transpose(A)", transpose(A), "\n")

            print("Vk ", Vk, "\n")
            print("wk ", wk, "\n")
            # V*_{k+1} wk = V*_{k+1} A V*_{k+1} = V*_{k+1} A V*_{k+1}
            # print("(conj.(transpose(Vk))*wk)[1] ", (conj.(transpose(Vk))*wk)[1], "\n")

            Hk[i,i] = (conj.(transpose(Vk))*wk)[1]

            if A != conj((transpose(A)))
                print("(conj.(transpose(Vk))*wk)[1] ", (conj.(transpose(vk))*Wk)[1], "\n")
                Hk[end, 1:end] = (conj.(transpose(vk))*Wk) # [1]
            end
            # Compute the largest eigenpar (theta,s) of H_{k+1} 
            # with the norm(s) = 1
            theta, eigenvector = eigs(Hk, which=:LM)
            s = eigenvector/norm(eigenvector) # normalized_eigenvector
            print("s ", s, "\n")
            
            u = Vk*s # Compute the Ritz vector u 
            print("u ", u, "\n")
            u_hat = A*u # Should also have: A*u = W_k*s

            print("u_hat ", u_hat, "\n")
            print("theta[1] ", theta[1], "\n")
            print("theta[1]*u ", theta[1]*u, "\n")
            r = u_hat - theta[1]*u # Residual vector 
            
        end
        print("norm of residual ", norm(r), "\n")
        if norm(r) <= tol
            print("Exited the loop using break")
            break 
        end 
        Vk = zeros(Float64, rows, cols)
        # Vk = Array{Float64}(undef, rows, cols)
        Vk[1:end,1] = vk
    
        Wk = zeros(Float64, rows, cols)
        # Wk = Array{Float64}(undef, rows, cols)
        Wk[1:end,1] = u_hat
        
        Hk = zeros(Float64, rows, cols)
        # Hk = Array{Float64}(undef, rows, cols)
        Hk[1,1] = theta[1]
    end
    return theta
end 

A = Array{Float64}(undef, 3, 3)
A[1,1] = 2
A[1,2] = -2
A[1,3] = 0
A[2,1] = -1
A[2,2] = 3
A[2,3] = -1 
A[3,1] = 0
A[3,2] = -1
A[3,3] = 4
# A = zeros(Int8, 3, 3)
# A[1,1] = 2
# A[1,2] = -1
# A[1,3] = 0
# A[2,1] = -1
# A[2,2] = 2
# A[2,3] = -1 
# A[3,1] = 0
# A[3,2] = -1
# A[3,3] = 2
print("A ", A, "\n")
# print(A[1,:])
# print("size(A) ", size(A)[2], "\n")

# print(eigen(A).values)

print(davidson_it(A), "\n")
end 