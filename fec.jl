include("graph.jl")

using LinearAlgebra, SparseArrays
using StatsBase

function SCF(G::Graph, sample_num::Int)
    n = G.n
    m = G.m
    d = degree_vector(G)
    in_forests = Vector{Bool}(undef, n)
    next = Vector{Int}(undef, n)
    root = Vector{Int}(undef, n)

    nbr, weight = neighbor_weight(G)
    diag = zeros(Float64, n)
    offdiag = zeros(Float64, m)
    ans = zeros(Float64, m)

    for _ in 1:sample_num
        fill!(in_forests, false)
        fill!(next, -1)
        fill!(root, 0)
        for src in 1:n
            u = src
            while in_forests[u] == false
                if rand(Float64) * (d[u] + 1) < 1
                    in_forests[u] = true
                    root[u] = u
                    diag[u] += 1 / sample_num
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

        for i in 1:m
            u = G.u[i]
            v = G.v[i]
            if root[u] == v
                offdiag[i] += 1 / sample_num
            end
        end
    end
    for i in 1:m
        u = G.u[i]
        v = G.v[i]
        ans[i] = (diag[u] + diag[v]) / offdiag[i] - 2
    end

    return ans
end

function SFQPlus(G::Graph, sample_num::Int)
    n = G.n
    m = G.m
    A = adjacency_matrix(G)
    d = degree_vector(G)
    in_forests = Vector{Bool}(undef, n)
    next = Vector{Int}(undef, n)
    root = Vector{Int}(undef, n)

    nbr, weight = neighbor_weight(G)

    diag = zeros(Float64, n)
    offdiag = zeros(Float64, m)
    ans = zeros(Float64, m)

    for u in 1:n
        diag[u] = 1 / (1 + d[u])
    end
    for _ in 1:sample_num
        fill!(in_forests, false)
        fill!(next, -1)
        fill!(root, 0)
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
                if A[r, u] != 0
                    diag[u] += 1 / (1 + d[u]) / sample_num
                end
                u = next[u]
            end
        end
        for i in 1:m
            u = G.u[i]
            v = G.v[i]
            k = root[u]
            if k == v
                offdiag[i] += 1 / (2 + d[v]) / sample_num
            elseif A[k, v] != 0
                offdiag[i] += 1 / (2 + d[v]) / sample_num
            end
        end
    end
    for i in 1:m
        u = G.u[i]
        v = G.v[i]
        ans[i] = (diag[u] + diag[v]) / offdiag[i] - 2
    end
    return ans
end

function IFGN(G::Graph, sample_num::Int)
    n = G.n
    m = G.m
    A = adjacency_matrix(G)
    d = degree_vector(G)
    in_forests = Vector{Bool}(undef, n)
    next = fill(-1, n)
    root = Vector{Int}(undef, n)

    nbr, weight = neighbor_weight(G)

    diag = zeros(Float64, n)
    offdiag = zeros(Float64, m)
    ans = zeros(Float64, m)

    for u in 1:n
        diag[u] = 1 / (1 + d[u])
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
        component = zeros(Int, n)
        for i in 1:n
            component[root[i]] += 1
        end
        for i in 1:n
            if component[i] == 0
                component[i] = component[root[i]]
            end
        end
        for i in 1:n
            for j in nbr[i]
                if root[i] == root[j]
                    diag[i] += 1 / (1 + d[i]) / component[i] / sample_num
                end
            end
        end
        for i in 1:m
            u = G.u[i]
            v = G.v[i]
            for k in nbr[v]
                if root[u] == root[k]
                    offdiag[i] += 1 / component[u] / (2 + d[v]) / sample_num
                end
            end
            if root[u] == root[v]
                offdiag[i] += 1 / component[u] / (2 + d[v]) / sample_num
            end
        end
    end
    for i in 1:m
        u = G.u[i]
        v = G.v[i]
        ans[i] = (diag[u] + diag[v]) / offdiag[i] - 2
    end
    return ans
end

function FECE(G::Graph, sample_num::Int)
    n = G.n
    m = G.m
    A = adjacency_matrix(G)
    d = degree_vector(G)
    in_forests = Vector{Bool}(undef, n)
    next = fill(-1, n)
    root = Vector{Int}(undef, n)

    nbr, weight = neighbor_weight(G)

    ans = zeros(Float64, m)
    num = zeros(Float64, m)

    for _ in 1:sample_num
        fill!(in_forests, false)
        fill!(next, -1)
        fill!(root, 0)
        for src in 1:n
            u = src
            while in_forests[u] == false
                if rand(Float64) * (d[u] + 1) < 1
                    in_forests[u] = true
                    root[u] = u
                    next[u] = -1
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

        component = zeros(Int, n)

        for i in 1:n
            component[root[i]] += 1
        end
        for i in 1:n
            if component[i] == 0
                component[i] = component[root[i]]
            end
        end

        for i in 1:m
            u = G.u[i]
            v = G.v[i]
            if root[u] == root[v]
                num[i] += 1 / component[u]
            end

            if root[u] != root[v]
                ans[i] += (1 / component[u] + 1 / component[v])
            end
        end
    end
    return ans ./ num
end
