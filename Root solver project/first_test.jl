using convexity,func,Pade_approx,peaks,Random_Matrix,root_finder

function First_test(x_start)
    xs = Any[x_start]
    ys = Any[f(x_start)]
    xr = x_start
    local xl = 0
    xn = x_start/2
    #check if our starting point gives us a value above 0
    ###### Need completing #####
    if ys[end] <= 0
        while ys[end] <=0
            x_start = 2*x_start
            push!(xs, x_start)
            push!(ys, f(x_start))
        end
    else #if our starting point was adequate, samples from right to left
        while ys[1] > 0
            insert!(xs,1,xn)
            insert!(ys,1,f(xn))
            xn = xn/2
        end
    end
    #one last check to see if it’s still above zero later on
    x_start = 1.5*x_start
    push!(xs, x_start)
    push!(ys, f(x_start))
    if ys[end]<0
        println("Need to check this scenario")
    end
    #check how many points we already have sampled to determine
    #how much more sampling needs to be done
    #(we’re going to do one sampling in between every already sampled points)
    xns = Any[]
    for i in range(1, length(xs)-1)
        push!(xns, (xs[i+1]+xs[i])/2)
    end
    for i in range(1, length(xns))
        for j in range(2, length(xs))
            if xns[i] < xs[j] && xns[i] > xs[j-1]
                insert!(xs, j, xns[i])
                insert!(ys,j,f(xns[i]))
            end
        end
    end
    #add a random point for test purposes (for the above test)
    # insert!(xs,3, 90)
    # insert!(ys,3,40)
    # insert!(xs,4, 95)
    # insert!(ys,4,70)
    #Use the padé to see if peaks are predicted in the area that is supposed to be convexity
    x_pade, y_pade = Pade(xs,ys)
    peaks = peak_finder(x_pade, y_pade)
    ########################
    ##### Some investigating needs to be done to determine what causes
    ##### convexity_test to give false positives when checking the padé_approx
    #### This issue doesnt arrise with Peak_finder
    # convexs1 = convexity_test(xs,ys)

    # convexs2 = convexity_test(x_pade, y_pade, true)
    # println(peaks)
    # println(convexs2)
    #determine which region is above the x axis
    above_region = xs
    for i in range(1,length(xs))
        if ys[i] < 0
        above_region = xs[i:end]
        end
    end
    #check if there is a predicted peak wihtin the "above" sampled region
    if isempty(peaks) == false
        for i in range(1, length(peaks))
            if peaks[i] > above_region[1] && peaks[i] < above_region[end]
                println("There is a predicted peak in the above region, do something about i \n")
            end
        end
    end
    # Determines the right and left bound in which more sampling might be needed
    xr = above_region[2]
    for i in range(1,length(xs))
        if ys[i] < 0
        xl = xs[i]
        end
    end
    return xs, ys, xl, xr
end
