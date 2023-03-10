module b_asym_only
export bv_asym_only
# creation of b 
function bv_asym_only(ei, l, l2, P) 
    # print("-ei/(2im) ", -ei/(2im), "\n")
    # print("(l[1]/(2im))*P*ei  ", (l[1]/(2im))*P*ei , "\n")

    # If we have a negative in front of the second term 
    # return (-((1+l[1])/(2im))+l[2]/2).*ei 

    val = -1/(2im)
    if length(l) > 0
        for i in eachindex(l)
            val -= l[i]/(2im)
        end 
    end 
    if length(l2) > 0
        for j in eachindex(l2)
            val += l2[j]/2
        end 
    end 
    return val.*ei
    
    # l[1]*
    # You can either divide by l[1] in b right here or multiply by l[1] in A in gmres


    # return .-ei/(2im) - (l[1]/(2im)).*ei # *P

    # At the end of the intership, we weren't sure if there was a negative
    # in front of the second term or not. 

    # If we don't have a negative in front of the second term 
    # return .-ei/(2im) + (l[1]/(2im)).*ei # *P

    # just like for A, the first lambda is associated 
    # with the asym part (l) and the second lambda is 
    # associated with the sym part (v)
end

end