function f(x)
    t = inv(A_0 + x*A_1)*(s_0+x*s_1)
    f_0 = 2*real(adjoint(t)*s_1) - adjoint(t)*A_1*t
    return f_0[1]