mutable struct Neuron{T<:AbstractFloat}
    w::Vector{Value{T}}
    b::Vector{Value{T}}
end

"""
    neuron(n::Int)
Construct a neuron with `n` inputs (features)
"""
function neuron(n::Int)
    w=value.(rand(n).*2 .-1)
    b=value.(rand(n).*2 .-1)
    Neuron(w,b)
end

# make a neuron a callable object
function (n::Neuron)(x)
    if length(x) != length(n.w)
        error("In calling n(x), expected $(length(n.w)) inputs to neuron, got $(length(x)):\n\tx = $(x)")
    end
    raw = sum(n.w .* x .+ n.b)
    out = tanh(raw)
    return out
end

function parameters(n::Neuron)
    return [n.w,n.b]
end



mutable struct Layer{T<:AbstractFloat}
    neurons::Vector{Neuron{T}}
    inputs::Int
    outputs::Int
end

"""
    layer(n_in::Int,n_out::Int)
Construct a layer of neurons. `n_in` is the number of inputs to each neuron in
the layer. `n_out` is the number of neurons in the layer (i.e. the number of
outputs of that layer)

layer(5,10) creates a layer with 10 neurons, each with 5 inputs.

"""
function layer(n_in::Int,n_out::Int)
    Layer([neuron(n_in) for _ in 1:n_out],n_in,n_out)
end

# make layer callable
function (l::Layer)(x)
    out = [n(x) for n in l.neurons]
    length(out) == 1 && return out[1]
    return out
end

function parameters(l::Layer)
    return parameters.(l.neurons)
end


mutable struct MLP{T<:AbstractFloat}
    layers::Vector{Layer{T}}
end

"""
    mlp(n_in,n_outs)
Construct a multi layer perceptron. `n_in` is the number of inputs to the MLP
and `n_outs` is a list of length `m` where `m` is the number of layers in the
MLP and `n_outs[i]` is the number of neurons in the `ith` layer.
"""
function mlp(n_in,n_outs)
    n_vec = [n_in;n_outs]
    MLP([layer(n_vec[i],n_vec[i+1]) for i in 1:length(n_outs)])
end

function (m::MLP)(x)
    for l in m.layers
        x = l(x)
    end
    return x
end

function parameters(m::MLP)
    return parameters.(m.layers)
end

function loss(y,y_pred,type)
    if type=="l2"
        l = sum((y.-y_pred).^2)
    elseif type=="l1"
        l = sum(abs(y.-y_pred))
    end
    return l
end