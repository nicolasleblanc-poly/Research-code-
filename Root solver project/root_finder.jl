#x1: starting point to the left
#x2 : starting point to the right
#: Maximum error we want between the actual zero and the approximated one
#N: Maximum number of iteration before returning a value
module root_finder 
export bissection 
function bissection(x1,x2,error,N)
    local fxm
    xm = (x1 + x2)/2
    compt = 0
    while abs(x2-x1)/(2*abs(xm)) > && compt < N
    xm = (x1 + x2)/2
    ans = Pade(big_x_sampled[end],big_y_sampled[end],rebuild_with=[x1,x2,xm])
    ys = ans[2]
    fx1 = ys[1]
    fx2 = ys[2]
    fxm = ys[3]
    if fx1*fxm < 0
    x2 = xm
    elseif fxm*fx2<0
    x1 = xm
    end
    compt +=1
    end
    return (xm, fxm)
end 
end 