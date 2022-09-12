export cheb2

function cheb2(n, a=-1.0, b=1.0)
    return [cos(pi*i/n) for i=0:n].*(b-a)/2.0 .+ (a+b)/2.0
end
