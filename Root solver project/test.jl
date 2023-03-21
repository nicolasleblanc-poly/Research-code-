using Plots, Distributed

###############################The function itself############################# 

A_0 = [-4.202369178258174 0.23656026179610068 0.9967817753194017 1.3621812143410215 0.7730591438014547 0.6795174235124107 1.201085140547982 1.4868076811790276; 0.23656026179610068 7.798410364498693 1.1887914375583606 0.8319234390137311 0.6993492068510417 1.3827615231298125 0.700451511387506 0.7056215986781617; 0.9967817753194017 1.1887914375583606 0.42299075325249924 0.29486161901242103 0.19378176034018868 1.000139458265159 0.9244827781178556 1.1508145100468172; 1.3621812143410215 0.8319234390137311 0.29486161901242103 0.01708738384810582 1.1788883010405404 0.3440989888794501 0.8452830180744944 1.031421035373452; 0.7730591438014547 0.6993492068510417 0.19378176034018868 1.1788883010405404 7.7279086959771925 0.5514991511284997 1.2655895299014635 0.4587696224505302; 0.6795174235124107 1.3827615231298125 1.000139458265159 0.3440989888794501 0.5514991511284997 -4.801460017525201 1.5136937331663745 1.5641726026980587; 1.201085140547982 0.700451511387506 0.9244827781178556 0.8452830180744944 1.2655895299014635 1.5136937331663745 -5.772356595393507 0.8863604625475946; 1.4868076811790276 0.7056215986781617 1.1508145100468172 1.031421035373452 0.4587696224505302 1.5641726026980587 0.8863604625475946 3.8057109577380377]  
A_1 = [2.868579595078885 2.0017974394942346 2.74322349984502 1.5384739564067065 1.2049901854320149 0.7844614927869878 0.5116454465287238 0.3954765714958245; 2.0017974394942346 2.1989032020338324 2.2633740613445834 1.860144380106243 1.5067979872002306 1.0776467254653175 0.49796391833968345 0.07010833916308676; 2.74322349984502 2.2633740613445834 3.481910164372317 2.2196452124024675 1.8506900180327013 1.308331972989803 0.9886660735458807 0.8016426250970934; 1.5384739564067065 1.860144380106243 2.2196452124024675 2.365509026271365 2.0482719098769335 1.1736822169267422 0.8666751085840902 0.6206594844699613; 1.2049901854320149 1.5067979872002306 1.8506900180327013 2.0482719098769335 2.5849404532937563 0.8654543936056173 1.1260900464597272 0.8144678880412133; 0.7844614927869878 1.0776467254653175 1.308331972989803 1.1736822169267422 0.8654543936056173 1.2779957150876138 0.6283583221225173 0.09773593867003194; 0.5116454465287238 0.49796391833968345 0.9886660735458807 0.8666751085840902 1.1260900464597272 0.6283583221225173 0.9344991358049112 0.616241342068419; 0.3954765714958245 0.07010833916308676 0.8016426250970934 0.6206594844699613 0.8144678880412133 0.09773593867003194 0.616241342068419 1.0923134840223656]
s_0 = [0.05622552393337055; 0.3954088760578657; 0.8193557283602528; 0.6414037922397975; 0.13623949797842694; 0.30045038855414574; 0.16677768125220171; 0.6697873147229706;;]
s_1 = [0.4991424981445124; 0.3705476346310461; 0.17111573451940554; 0.5073875443016921; 0.42246573217942185; 0.43721333864318745; 0.937501475360908; 0.5148246209305662;;]

function f(x)
    t = inv(A_0 + x*A_1)*(s_0+x*s_1)
    f_0 =  2*real(adjoint(t)*s_1) - adjoint(t)*A_1*t 
    return f_0[1]
end

# Plots the piecewise function
xmax = 100
xmin = 0.0
N = 200
px = [i for i in range(xmin, xmax, N)]
py = map(f, px)
ymax = 50
ymin = -200
lbound = 0
plt1 = plot(px,py, ylim=(ymin,ymax), legend = false)
display(plt1)
sleep(3)

##########################Peak finders#########################################
function peak_finder(xs, ys)  # Finds the peaks within the sampled point 
    slope_criteria = 2 # This needs to be more dynamic, to account for the 
    # amount of sampled points
    peaks = []
    slopes = [] 
    for i in range(2, length(xs))
        added = false #verify if a peak has been added twice
        slope =  (ys[i]-ys[i-1])/(xs[i]-xs[i-1])
        push!(slopes, slope)

        # println(i)
        # println(slope)
        # println(slopes)
        # Test : This one seems like a bad test to conduct, doesnt narrow down 
        # the position enough
        # if slope > slope_criteria
        #     push!(peaks,xs[i])
        #     # println(slope)
        #     # println(" ")
        # end

        # Test 1: Checks a sign change in the slope
        if i > 2
            
            if (slopes[i-2]\slopes[i-1]) < 0 && ys[i-2] < -5 # Change of sign of 
                # the slope, implying a peak #the ys[i] <-5 is to prevent 
                # finding regions between the peaks
                push!(peaks, xs[i-1]) #note: we take xs[i-1] instead of xs[i-2] 
                # because the i-1 value is to the right of the peak
                added = true
            end
        end
        # Test2 : Checks if the slope before and after a point stopped growing 
        # (indicating that there was a peak)
        if i > 3
            if abs(slopes[i-3]) < abs(slopes[i-2]) && abs(slopes[i-2]) > abs(slopes[i-1]) && added == false && ys[i] <-5 #the ys[i] <-5 is to prevent finding regions between the peaks
                push!(peaks, xs[i-1])
            end
        end
    end
    return peaks
end

# # OG version of the code 
# ################################################### Padé function
# # x : x values for the sampled points
# # y : y values for the sampled points
# # N : number of points used to evaluate the approximate once it has been built using x,y 
# # xl : left bound from which the Padé is evaluated 
# # xr : right bound up to which the Padé is evaluated
# # rebuild_with : evaluates the constructed Padé at the x values in this list

# function Pade(x, y; N = 500, xl = 0.0, xr = xmax, rebuild_with = [])
#     #Padé approximant algorithm
#     x = x
#     r = y
#     l =  length(x)
#     R= zeros(l)
#     X= zeros(l)
#     P= zeros(l)
#     S= zeros(l)
#     M= zeros(l)

#     for i in range(1,l) #first loop 
#         R[i] = r[i]
#         X[i] = x[i]
#     end

#     for j in range(1,l)#second loop
#         P[j] = R[j]
#         for s in range(j+1, l)
#             S[s] = R[j]
#             R[s] = R[s] - S[s]
#             if R[s] == 0
#                 #println("Huston, we have a problem, ABORT.")
#             else
#                 M[s] = X[s] - X[j]
#                 R[s] = M[s]/R[s]
#                 if j-1 == 0 # to avoid indexing at the P[0] position
#                     R[s] = R[s]
#                 else
#                     R[s] = R[s] + P[j-1]
#                 end
#             end
#         end
#     end

#     function rebuild(x)  #rebuild the approximation from the little blocks
#         A = zeros(l)
#         B = zeros(l)
#         A[1] = P[1]
#         B[1] = 1
#         print("P[1] ", P[1], "\n")
#         print("P[2] ", P[2], "\n")
#         print("x ", x, "\n")
#         print("X[1] ", X[1], "\n")
#         A[2] = P[2]*P[1] + (x-X[1])
#         B[2] = P[2]
#         for i in range(2, l-1)
#             A[i+1] = A[i]*(P[i+1] -P[i-1]) + A[i-1]*(x-X[i])
#             B[i+1] = B[i]*(P[i+1] -P[i-1]) + B[i-1]*(x-X[i])
#         end
#         if isinf(abs(A[l])) == true || isinf(abs(B[l])) == true || isnan(abs(A[l])) == true || isnan(abs(B[l])) ==true
#             throw(Error) #not sure what to do when this happens yet, problems occur when N exceeds 336
#         else
#             return A[l]/B[l]
#         end
#     end
#     local px
#     if isempty(rebuild_with)== true
#         px = [i for i in range(xl, xr, N)]
#         approx = map(rebuild, px)
#     else
#         px = rebuild_with
#         approx = map(rebuild, px)
#     end
#     return (px, approx)
# end

# My version 
#########################Padé function######################################### 
# x : x values for the sampled points
# y : y values for the sampled points
# N : Number of points used to evaluate the approximate once it has been built 
# using x,y 
# xl : Left bound from which the Padé is evaluated 
# xr : Right bound up to which the Padé is evaluated
# rebuild_with : Evaluates the constructed Padé at the x values in this list

function Pade(x, y; N = 500, xl = 0.0, xr = xmax, rebuild_with = [])
    # Padé approximant algorithm
    x = x
    r = y
    l = length(x)
    R = zeros(l)
    X = zeros(l)
    P = zeros(l)
    S = zeros(l)
    M = zeros(l)

    for i in range(1,l) # First loop 
        R[i] = r[i]
        X[i] = x[i]
    end

    for j in range(1,l) # Second loop
        P[j] = R[j]
        for s in range(j+1, l)
            S[s] = R[j]
            R[s] = R[s] - S[s]
            if R[s] == 0
                println("Houston, we have a problem, ABORT.")
            else
                M[s] = X[s] - X[j]
                R[s] = M[s]/R[s]
                if j-1 == 0 # To avoid indexing at the P[0] position
                    R[s] = R[s]
                else
                    R[s] = R[s] + P[j-1]
                end
            end
        end
    end

    local px
    if isempty(rebuild_with)== true
        print("Sampling \n")
        px = [i for i in range(xl, xr, N)]
        approx = zeros(length(px))
        # approx = map(rebuild, px, l, P)
        for i in eachindex(px)
            approx[i] = rebuild(px[i], l, P, X)
        end 
    else
        print("Don't need to resample everthing \n")
        px = rebuild_with
        print("px ", px, "\n")
        # approx = map(rebuild, px, l, P)
        for i in eachindex(x)
            approx[i] = rebuild(px[i], l, P, X)
        end
    end
    return (px, approx)
end

function rebuild(x,l,P,X)  # Rebuild the approximation from the little blocks
    print("x ", x, "\n")
    print("l ", l, "\n")
    print("P ", P, "\n")
    print("X ", X, "\n")

    A = zeros(l)
    B = zeros(l)
    A[1] = P[1]
    B[1] = 1

    # print("P[1] ", P[1], "\n")
    # print("P[2] ", P[2], "\n")
    # print("x ", x, "\n")
    # print("X[1] ", X[1], "\n")

    A[2] = P[2]*P[1] + (x-X[1])
    B[2] = P[2]
    for i in range(2, l-1)
        A[i+1] = A[i]*(P[i+1] -P[i-1]) + A[i-1]*(x-X[i])
        B[i+1] = B[i]*(P[i+1] -P[i-1]) + B[i-1]*(x-X[i])
    end
    if isinf(abs(A[l])) == true || isinf(abs(B[l])) == true || isnan(abs(A[l])) == true || isnan(abs(B[l])) ==true
        print("Problem \n")
        throw(Error) # Not sure what to do when this happens yet, 
        # problems occur when N exceeds 336
    else
        return A[l]/B[l]
    end
end

###################First test that needs to be done############################ 
# This test samples the function initially to determine where is the
# region in which the function is definitly above zero

function First_test(x_start)
    xs = Any[x_start]
    ys = Any[f(x_start)]
    xr = x_start
    local xl = 0
    xn = x_start/2
    # Check if our starting point gives us a value above 0
    ###### Need compleating #####
    if ys[end] <= 0
        while ys[end] <=0
            x_start = 2*x_start
            push!(xs, x_start)
            push!(ys, f(x_start))
        end
    else # If our starting point was adequate, samples from right to left
        while ys[1] > 0
            insert!(xs,1,xn)
            insert!(ys,1,f(xn))
            xn = xn/2
        end
    end
    # One last check to see if it's still above zero later on
    x_start = 1.5*x_start
    push!(xs, x_start)
    push!(ys, f(x_start))
    if ys[end]<0
        println("Need to check this scenario")
    end
    # Check how many points we already have sampled to determine 
    # how much more sampling needs to be done 
    # (we're going to do one sampling in between every already sampled points)
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
    # Add a random point for test purposes (for the above test)
    # insert!(xs,3, 90)
    # insert!(ys,3,40)
    # insert!(xs,4, 95)
    # insert!(ys,4,70)
    # Use the padé to see if peaks are predicted in the area that is supposed to be convexity
    print("xs ", xs, "\n")
    print("ys ", ys, "\n")
    
    x_pade, y_pade = Pade(xs,ys)
    # print("x_pade ", x_pade, "\n")
    # print("y_pade ", y_pade, "\n")
    peaks = peak_finder(x_pade, y_pade)

    ########################
    ##### Some investigating needs to be done to determine what causes
    ##### convexity_test to give false positives when checking the padé_approx
    ####  This issue doesnt arrise with Peak_finder
    # convexs1 = convexity_test(xs,ys)
    # convexs2 = convexity_test(x_pade, y_pade, true)
    # println(peaks)
    # println(convexs2)

    # Determine which region is above the x axis
    above_region = xs
    for i in range(1,length(xs))
        if ys[i] < 0
            above_region = xs[i:end]
        end
    end
    # Check if there is a predicted peak wihtin the "above" sampled region
    if isempty(peaks) == false
        for i in range(1, length(peaks))
            if peaks[i] > above_region[1] && peaks[i] < above_region[end]
                println("There is a predicted peak in the above region, do something about it")
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

# xl = 0
# xr = 2000 # The search is gonna start here
# xs, ys , xl, xr = First_test(xr)
# global x_sampled = xs
# global y_sampled = ys
# global compt = 0

# print("xs ", xs, "\n")
# print("ys ", ys, "\n")
# print("xl ", xl, "\n")
# print("xr ", xr, "\n")


###########################Bissection root finding#############################
"""
The code was modified to access the Padé approximant constructed from
the already sampled points. The issue is that for every evaluation of the
function, the Padé approximant is rebuilt and reevaluated. This is because
of the way the Padé approximant code was written. This could be made
much more efficient.
"""
# TO DO : ______________
# Could be rewritten better, modified it to work with the Padé approx
################################
################################
# x1: starting point to the left
# x2 : starting point to the right
# err: Maximum error we want between the actual zero and the approximated one
# N: Maximum number of iteration before returing a value
function bissection(xs,ys,x1,x2,err,N)
    local fxm
    # big_x_sampled = Any[]
    # big_y_sampled = Any[]
    # push!(big_x_sampled,xs)
    # push!(big_y_sampled,ys)

    xm = (x1 + x2)/2
    compt = 0
    while abs(x2-x1)/(2*abs(xm)) > err && compt < N
        xm = (x1 + x2)/2
        # ans = Pade(big_x_sampled[end],big_y_sampled[end],rebuild_with=[x1,x2,xm])
        ans = Pade(xs[end],ys[end],rebuild_with=[x1,x2,xm])
        ys = ans[2]
        fx1 = ys[1]
        fx2 = ys[2]
        fxm = ys[3]
        if  fx1*fxm < 0
            x2 = xm
        elseif fxm*fx2<0
            x1 = xm
        end
        compt +=1
    end
    return (xm, fxm)
end

# xl = 0
xr = 2000 # The search is gonna start here
xs, ys , xl, xr = First_test(xr)
err = 1e-6
N = 500
bissect = bissection(xs,ys,xl,xr,err,N)
print("bissection test ", bissect, "\n")

# TBD if we really need to modify the bissection method to do like it says below
# # - Modify the bissection method algorithm to work with the sampled points ✓
# #   instead of the function. ✓
# function bissection_sampled(x1,x2,err,N)
#     local fxm
#     xm = (x1 + x2)/2
#     compt = 0
#     while abs(x2-x1)/(2*abs(xm)) > err && compt < N
#         xm = (x1 + x2)/2
#         # If we use the Pade function 
#         # ans = Pade(big_x_sampled[end],big_y_sampled[end],rebuild_with=[x1,x2,xm])
#         # If we use the sampled points 
#         ans = 
#         ys = ans[2]
#         fx1 = ys[1]
#         fx2 = ys[2]
#         fxm = ys[3]
#         if  fx1*fxm < 0
#             x2 = xm
#         elseif fxm*fx2<0
#             x1 = xm
#         end
#         compt +=1
#     end
#     return (xm, fxm)
# end


# This function is used farther below 
############################Padé stop criteria test###########################
# This function checks wheter that last sampling changed the Padé approximant
# significantly or not. If not, stop sampling.
# It keeps in memory previous sampling to compare them
# Should it dump some of them (would that save memory or once it's assigned it's too late?)
global big_x_sampled = Any[]
global big_y_sampled = Any[]
global errors = Any[]
global ratios = Any[]
global stop_crit = false

function pade_stop_criteria(xs,ys,xl)
    local pade_x1, pade_y1, pade_x2, pade_y2, error
    global big_x_sampled
    global big_y_sampled
    global errors
    global stop_crit
    push!(big_x_sampled,xs)
    push!(big_y_sampled,ys)
    # println(big_x_sampled)
    # println(big_y_sampled)
    # println("xl =", xl)
    if length(big_x_sampled) <= 1
        # Do nothing
    elseif length(big_x_sampled) > 1
        # Compare samplings
        N = 1000
        pade_x1, pade_y1 = Pade(big_x_sampled[end],big_y_sampled[end],N=N,xl =xl)
        pade_x2, pade_y2 = Pade(big_x_sampled[end-1],big_y_sampled[end-1],N=N,xl =xl)
        # println(pade_x1)
        # println(pade_x2)
        # Calculate error between the two Padé's in between a relevant x range
        error = 0
        for i in range(1,length(pade_x1))
            error += abs(pade_y2[i] - pade_y1[i])/N
        end
        push!(errors, error)
        # Critera to see by how much the error shrunk from one iteration to the next
        if length(errors)>1
            ratio = errors[end-1]/errors[end]
            push!(ratios, ratio)
            # Checks if the error is diminishing with each extra sampling (needs to diminish twice in a row)
            if length(ratios)>2
                if ratios[end]<ratios[end-1] && ratios[end-1] < ratios[end-2]
                    stop_crit = true
                    println("done")
                end
            end
        end
    end
    
end
# print(pade_stop_criteria(xs,ys,xl))


xl = 0
xr = 2000 #the search is gonna start here
xs, ys , xl, xr = First_test(xr)
global x_sampled = xs
global y_sampled = ys
global compt = 0


function big_boiii(display_plot = false)
    local plt
    if display_plot ==true
        plt = plot(x_sampled,y_sampled, markershape = :circle, color = :green, 
        linewidth = 0, legend = false, ylim =(ymin,ymax),xlim =(xmin,xmax))

        plot!(px,py)
    end
    # The stopping critera is that the error between padés need to diminish
    # for two consecutive samplings
    while stop_crit == false
        global xs, ys, compt
        local index 
        index = 1
        pade_x, pade_y = Pade(x_sampled, y_sampled, N =5000)
        peaks = peak_finder(pade_x, pade_y)
        println(peaks)
        # Sample near the peak but far enough from last sampled point (inbetween both values)
        # lastpeak = peaks[end]
        # plastpeak = peaks[end-1]
        # next = 0
        # #### Finds what sampled point is closest to the suspected peak
        # #### in order to sample close to it (inbetween)
        # for i in x_sampled
        #     if i > lastpeak
        #         next = i
        #         break
        #     end
        # end

        # #### This needs cleaning up (xn1, xn2,xn3), might be too much to sample 3 times each round
        # #### Also these variables names are confusing
        # xn1 = (lastpeak+next)/2 # This needs cleaning up
        # xn2 = (plastpeak+next)/2
        # xn3 = lastpeak
        # xs = [xn3, xn1]
        # ys = [f(xn3),f(xn1)]        
        xn = last(peaks)
        xs = [xn]
        ys = [f(xn)]

        ###### Adds the new sampled values in order to the already sampled ones (in ascending x order)
        ###### This could be made into it's own function ...
        for i in range(1, length(xs))
            for j in range(2, length(x_sampled))
                if xs[i] < x_sampled[j] && xs[i] > x_sampled[j-1]
                    insert!(x_sampled, j, xs[i])
                    insert!(y_sampled,j,ys[i])
                    index = j
                end
                if xs[i] < x_sampled[j] && j ==2
                    insert!(x_sampled, 1, xs[i])
                    insert!(y_sampled,1,ys[i])
                    index = j
                end
                if xs[i] > x_sampled[j] && j==length(x_sampled)
                    insert!(x_sampled, j+1, xs[i])
                    insert!(y_sampled,j+1,ys[i])
                    index = j
                end
            end
        end
        println(xn)
        println(x_sampled)
        # Checks if this sampling has changed something about our evaluation
        pade_stop_criteria(x_sampled[index:end], y_sampled[index:end],xn)
        # Plots the different padés (if needed)
        if display_plot==true
            for i in range(1,length(big_x_sampled))
                local pade_x, pade_y
                N=5000
                pade_x, pade_y = Pade(big_x_sampled[i],big_y_sampled[i],N=N,xl = xn)
                plot!(pade_x,pade_y)
                plot!(x_sampled,y_sampled, markershape = :circle, linewidth = 0, color = :green)
            end
            display(plt)
            sleep(1)
        end
        compt+=1
    end

    # We can narrow the position of the zero using our sampled points
    println(x_sampled)
    for i in range(1, length(x_sampled))
        sleep(1)
        # println(x_sampled[end-i+1])
        if y_sampled[i]<0
            xl = y_sampled[i]
        end
        if y_sampled[end-i+1] < 0
            println(x_sampled[end-i+1])
            xr = x_sampled[end-i+2]
        end
    end
    println(xl," ", xr)
    # Finds the zero using the last Padé approximant (bissection method)
    zeros = bissection(xl,xr,10^(-10),100)
    # println("The real root is : ", 65.52353857114213)
    return zeros[1]
end

# println(x_sampled)
# ans = big_boiii(true)
# println("The guessed root is: ", ans)

