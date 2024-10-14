include("graph.jl")

using LinearAlgebra, SparseArrays
using StatsBase, Laplacians

function SCF(G::Graph, sample_num::Int)
    n = G.n
    d=degree_vector(G)
    in_forests = Vector{Bool}(undef,n)
    next = Vector{Int}(undef,n)
    root = Vector{Int}(undef,n)

    nbr,weight=neighbor_weight(G)
    ans = zeros(Float64, n)
    for _ in 1:sample_num
        fill!(in_forests, false)
        for src in 1:n
            u = src
            while in_forests[u] == false
                if rand(Float64) * (d[u] + 1) < 1
                    in_forests[u] = true
                    root[u] = u
                    ans[u]+=1/sample_num
                    break
                end
                next[u] = StatsBase.sample(nbr[u], Weights(weight[u]))
                u = next[u]
            end
            r = root[u]
            u = src
            while in_forests[u] == false
                in_forests[u] = true
                root[u] = r
                u = next[u]
            end
        end
    end
    return ans
end

function SFQPlus(G::Graph, sample_num::Int)
    n = G.n
    A=adjacency_matrix(G)
    d=degree_vector(G)
    in_forests = Vector{Bool}(undef,n)
    next = Vector{Int}(undef,n)
    root = Vector{Int}(undef,n)

    nbr,weight=neighbor_weight(G)
    ans = zeros(Float64, n)
    for u in 1:n
        ans[u]=1/(1+d[u])
    end
    for _ in 1:sample_num
        fill!(in_forests, false)
        for src in 1:n
            u = src
            while in_forests[u] == false
                if rand(Float64) * (d[u] + 1) < 1
                    in_forests[u] = true
                    root[u] = u
                    break
                end
                next[u] = StatsBase.sample(nbr[u], Weights(weight[u]))
                u = next[u]
            end
            r = root[u]
            u = src
            while in_forests[u] == false
                in_forests[u] = true
                root[u] = r
                if A[r,u]!=0
                    ans[u]+=1/(1+d[u])/sample_num
                end
                u = next[u]
            end
        end
    end
    return ans
end

function IFG(G::Graph, sample_num::Int)
    n = G.n
    A=adjacency_matrix(G)
    d=degree_vector(G)
    in_forests = Vector{Bool}(undef,n)
    next = Vector{Int}(undef,n)
    root = Vector{Int}(undef,n)

    nbr,weight=neighbor_weight(G)
    ans = zeros(Float64, n)
    for _ in 1:sample_num
        fill!(in_forests, false)
        for src in 1:n
            u = src
            while in_forests[u] == false
                if rand(Float64) * (d[u] + 1) < 1
                    in_forests[u] = true
                    root[u] = u
                    break
                end
                next[u] = StatsBase.sample(nbr[u], Weights(weight[u]))
                u = next[u]
            end
            r = root[u]
            u = src
            while in_forests[u] == false
                in_forests[u] = true
                root[u] = r
                u = next[u]
            end
        end

        component=zeros(Int,n)
        for i in 1:n
            component[root[i]]+=1
        end
        for i in 1:n
            if component[i]==0
                component[i]=component[root[i]]
            end
        end
        for i in 1:n
            ans[i]+=1/component[i]/sample_num
        end
    end
    return ans
end

function IFGN(G::Graph, sample_num::Int)
    n = G.n
    A=adjacency_matrix(G)
    d=degree_vector(G)
    in_forests = Vector{Bool}(undef,n)
    next=fill(-1,n)
    root = Vector{Int}(undef,n)

    nbr,weight=neighbor_weight(G)
    ans = zeros(Float64, n)

    for u in 1:n
        ans[u]=1/(1+d[u])
    end

    for _ in 1:sample_num
        fill!(in_forests, false)
        for src in 1:n
            u = src
            while in_forests[u] == false
                if rand(Float64) * (d[u] + 1) < 1
                    in_forests[u] = true
                    root[u] = u
                    break
                end
                next[u] = StatsBase.sample(nbr[u], Weights(weight[u]))
                u = next[u]
            end
            r = root[u]
            u = src
            while in_forests[u] == false
                in_forests[u] = true
                root[u] = r
                u = next[u]
            end
        end
        component=zeros(Int,n)
        for i in 1:n
            component[root[i]]+=1
        end
        for i in 1:n
            if component[i]==0
                component[i]=component[root[i]]
            end
        end
        for i in 1:n
            for j in nbr[i]
                if root[i]==root[j]
                    ans[i]+=1/(1+d[i])/component[i]/sample_num
                end
            end
        end
    end
    return ans
end