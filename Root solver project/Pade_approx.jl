module Pade_approx 
export Pade, rebuild 
function Pade(x, y; N = 500, xl = 0.0, xr = xmax, rebuild_with = [])
    #Padé approximant algorithm
    x = x
    r = y
    l = length(x)
    R= zeros(l)
    X= zeros(l)
    P= zeros(l)
    S= zeros(l)
    M= zeros(l)
    for i in range(1,l) #first loop
        R[i] = r[i]
        X[i] = x[i]
    end
    for j in range(1,l)#second loop
        P[j] = R[j]
        for s in range(j+1, l)
            S[s] = R[j]
            R[s] = R[s] - S[s]
            if R[s] == 0
            #println("Houston, we have a problem, ABORT.")
            else
                M[s] = X[s] - X[j]
                R[s] = M[s]/R[s]
                if j-1 == 0 # to avoid indexing at the P[0] position
                    R[s] = R[s]
                else
                    R[s] = R[s] + P[j-1]
                end
            end
        end
    end
    
    local px
    if isempty(rebuild_with)== true
        px = [i for i in range(xl, xr, N)]
        approx = map(rebuild, px)
    else
        px = rebuild_with
        approx = map(rebuild, px)
    end
    return (px, approx)
end

function rebuild(x) #rebuild the approximation from the little blocks
    A = zeros(l)
    B = zeros(l)
    A[1] = P[1]
    B[1] = 1
    A[2] = P[2]*P[1] + (x-X[1])
    B[2] = P[2]
    for i in range(2, l-1)
        A[i+1] = A[i]*(P[i+1] -P[i-1]) + A[i-1]*(x-X[i])
    B[i+1] = B[i]*(P[i+1] -P[i-1]) + B[i-1]*(x-X[i])
    end
    if isinf(abs(A[l])) == true || isinf(abs(B[l])) == true || isnan(abs(A[l])) == true
        throw(Error) #not sure what to do when this happens yet, problems occur when N e
    else
        return A[l]/B[l]
    end
end

end 