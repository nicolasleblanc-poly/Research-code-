

function convexity_test(x,y, display_plot = false)
    local xl = 0
    local i = 1
    while i < length(x) #while loop to easily skip iteration once we’ve found a point of non
        for j in range(i+2, length(x))
            for k in range(i+1,j-1)
                p1 = (x[i], y[i])
                p2 = (x[k], y[k])
                p3 = (x[j], y[j])
                slope = (p3[2] - p1[2])/(p3[1] - p1[1])
                intercept = p1[2] - slope*(p1[1])
                Y = slope*p2[1] + intercept
                if display_plot == true
                    small_x = [i for i in range(p1[1], p3[1],10)]
                    small_y = [(slope*i + intercept) for i in small_x]
                    plt = plot(px,py, ylim=(ymin,ymax), legend = false)
                    #plot!(x,y, markershape = :circle, color = :black)  
                    plot!(small_x, small_y)
                    plot!([p1[1]], [p1[2]], markershape = :circle, color = :green)
                    plot!([p3[1]], [p3[2]], markershape = :circle, color = :blue)
                    plot!([p2[1]], [p2[2]], markershape = :circle, color = :red)
                    display(plt)#necessary to display the graph (equivalent to plt.show() in
                    #sleep(0.1) #delay to allow for the graph to display
                end
                if Y > p2[2]
                    println("sampled point ",p2[2])
                    println("pedicted point ", Y)
                    println("____________")
                    xl = x[k]
                    i+=1
                end
            end
        end
        i+=1
    end
    return xl
end
