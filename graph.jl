using LinearAlgebra
using SparseArrays
using Laplacians

struct Graph
    n :: Int 
    m :: Int 
    u :: Vector{Int}
    v :: Vector{Int}
    w :: Vector{Float32}
end

function read_data(filename::AbstractString)
    lines = readlines("data/input/$filename")
    n, m = 0, 0
    u, v, w = Int[], Int[], Float32[]
	mapping = Dict{Int, Int}()
	current_index = 1
	for line in lines
		if line[1] == '#' || line[1] == '%'
			continue
		end
		uu = parse(Int, split(line)[1])
		vv = parse(Int, split(line)[2])
		if !haskey(mapping, uu)
			mapping[uu] = current_index
			current_index += 1
		end
		if !haskey(mapping, vv)
			mapping[vv] = current_index
			current_index += 1
		end
		push!(u, mapping[uu])
		push!(v, mapping[vv])
		push!(w, 1)
		m+=1
	end
	n = length(mapping)
    return Graph(n, m, u, v, w)
end

function degree_matrix(G)
	u=zeros(G.n);
	d=zeros(G.n);
	for i=1:G.n
		u[i]=i
	end
	for i=1:G.m
		if G.w[i]>0
			d[G.u[i]]+=G.w[i];
			d[G.v[i]]+=G.w[i];
		elseif G.w[i]<0
			d[G.u[i]]-=G.w[i];
			d[G.v[i]]-=G.w[i];
		end
	end
	return sparse(u,u,d)
end

function degree_vector(G)
	d=zeros(G.n);
	for i=1:G.m
		if G.w[i]>0
			d[G.u[i]]+=G.w[i];
			d[G.v[i]]+=G.w[i];
		elseif G.w[i]<0
			d[G.u[i]]-=G.w[i];
			d[G.v[i]]-=G.w[i];
		end
	end
	return d
end

function adjacency_matrix(G)
	u=zeros(2*G.m);
	v=zeros(2*G.m);
	w=zeros(2*G.m);
	for i=1:G.m
		u[i]=G.u[i];
		v[i]=G.v[i];
		w[i]=G.w[i];
		u[G.m+i]=G.v[i];
		v[G.m+i]=G.u[i];
		w[G.m+i]=G.w[i];
	end
	return sparse(u,v,w)
end

function neighbor_weight(G::Graph)
	n = G.n
	m = G.m
	nbr = Array{Array{Int, 1}}(undef, n)
	weight = Array{Array{Float64, 1}}(undef, n)
	for i in 1:n
		nbr[i] = Int[]
		weight[i] = Float64[]
	end
	for i in 1:m
		push!(nbr[G.u[i]], G.v[i])
		push!(nbr[G.v[i]], G.u[i])
		push!(weight[G.u[i]], abs(G.w[i]))
		push!(weight[G.v[i]], abs(G.w[i]))
	end
	return nbr, weight
end

