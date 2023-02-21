mutable struct Neuron{T<:AbstractFloat}
    w::Vector{Value{T}}
    b::Vector{Value{T}}
end

function neuron(n)
    w=value.(rand(n).*2 .-1)
    b=value.(rand(n).*2 .-1)
    Neuron(w,b)
end

function (n::Neuron)(x)
    raw = sum(n.w .* x .+ n.b)
    out = tanh(raw)
    return out
end


mutable struct Layer{T<:AbstractFloat}
    neurons::Vector{Neuron{T}}
end

function layer(n_in,n_out)
    Layer([neuron(n_in) for _ in 1:n_out])
end

function (l::Layer)(x)
    out = [n(x) for n in l.neurons]
    length(out) == 1 && return out[1]
    return out
end


mutable struct MLP{T<:AbstractFloat}
    layers::Vector{Layer{T}}
end

function mlp(n_in,n_outs)
    MLP([layer])