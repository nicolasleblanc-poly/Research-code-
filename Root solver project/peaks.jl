module peaks
export peak_finder
function peak_finder(xs, ys) #finds the peaks within the sampled point
    slope_criteria = 2 #this needs to be more dynamic, to account for the amount of sampled
    peaks = []
    slopes = []
    for i in range(2, length(xs))
        added = false #verify if a peak has been added twice
        slope = (ys[i]-ys[i-1])/(xs[i]-xs[i-1])
        push!(slopes, slope)

        # println(i)
        # println(slope)
        # println(slopes)
        # Test : This one seems like a bad test to conduct, doesnt narrow down the position
        # if slope > slope_criteria
            # push!(peaks,xs[i])
        #   # println(slope)
        #   # println(" ")
        # end

        # Test 1: Checks a sign change in the slope
        if i > 2
            if (slopes[i-2]\slopes[i-1]) < 0 && ys[i-2] < -5 #change of sign of the slope, i
                push!(peaks, xs[i-1]) #note: we take xs[i-1] instead of xs[i-2] because the
                added = true
            end
        end
        # Test2 : Checks if the slope before and after a point stopped growing (indicating th
        if i > 3
            if abs(slopes[i-3]) < abs(slopes[i-2]) && abs(slopes[i-2]) > abs(slopes[i-1]) &&
                push!(peaks, xs[i-1])
            end
        end
    end
    return peaks
end
end 