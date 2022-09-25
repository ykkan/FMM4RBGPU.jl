export maxlevel
export ncluster
export nodeindexrangeat
export leafindexrange

function maxlevel(N::I, N0::I) where {I}
    return (N > N0) ? ceil(I, log(N/N0)/log(2)) : zero(I)
end

function ncluster(N::I, N0::I) where {I}
    ml = maxlevel(N,N0)
    return 2^(ml+1)-1
end

function ncluster(ml::I) where {I}
    return 2^(ml+1)-1
end

function nodeindexrangeat(l)
    return (2^l):(2^(l+1)-1)
end

function leafindexrange(N, N0)
    ml = maxlevel(N,N0)
    return nodeindexrangeat(ml)
end
