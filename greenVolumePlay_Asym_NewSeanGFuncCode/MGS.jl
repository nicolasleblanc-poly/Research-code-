using LinearAlgebra, Random, Arpack

function MGS(V, t) # Modified_gram_schmidt
    # vk = 0
    print("V ", V, "\n")
    for j = 1:size(V)[2]
        vj = V[1:end,j]
        print("vj ", vj, "\n")
        global qj = vj/norm(vj)
        print("qj ", qj, "\n")
        print("(conj.(transpose(qj))*vk) ", (conj.(transpose(qj))*vj)[1], "\n")
        # for k = 1:j-1
        # for k = j+1:i
        for k = j+1:size(V)[2]
            print("(conj.(transpose(qj))*vk)*qj ", (conj.(transpose(qj))*vj)[1]*qj, "\n")
            # vj -= ((conj.(transpose(qj))*vj)[1])*qj
            print("vj ", vj, "\n")
            print("((conj.(transpose(qj))*vj)[1])*qj ", ((conj.(transpose(qj))*vj)[1])*qj, "\n")
            vj = vj - ((conj.(transpose(qj))*vj)[1])*qj
            # vk = vk - (conj.(transpose(vk))*qj)*qj
        end
        print("dot product 1 ", dot(vj,qj),"\n")
    end
    t = t-((conj.(transpose(qj))*t)[1])*qj
    print("dot product 2 ", dot(t,qj),"\n")

    # print("product ", conj.transpose(vk), t,"\n")
    return t = t-((conj.(transpose(qj))*t)[1])*qj
    # return t = t-(conj.(transpose(t))*vk)*vk
end 

A = Array{Float64}(undef, 3, 2)
# A = zeros(Int8, 3, 3)
A[1,1] = 1
A[2,1] = 2
A[3,1] = 2
A[1,2] = -1
A[2,2] = 0
A[3,2] = 2
# A[1,3] = 0
# A[2,3] = 0 
# A[3,3] = 1
print("A ", A, "\n")

t = Array{Float64}(undef, 3, 1)
# A = zeros(Int8, 3, 3)
t[1,1] = 0
t[2,1] = 0
t[3,1] = 1

print("MGS ", MGS(A,t) )
