xcells=ycells=zcells=9
l_sum = 3
for i=0:l_sum-1
    print("i ", i, "\n")

    print("i*xcells ", i*xcells/l_sum+1, "\n")
    print("(i+1)*xcells ", (i+1)*xcells/l_sum, "\n")

    print("i*ycells ", i*ycells/l_sum+1, "\n")
    print("(i+1)*ycell ", (i+1)*ycells/l_sum, "\n")

    print("i*zcells ", i*zcells/l_sum+1, "\n")
    print("(1+i)*zcells ", (1+i)*zcells/l_sum, "\n")
end 