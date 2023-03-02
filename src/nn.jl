mutable struct Neuron{T<:AbstractFloat,F<:Union{Function,Nothing}}
    w::Vector{Value{T}}
    b::Value{T}
    activation::F
end

"""
    neuron(n::Int)
Construct a neuron with `n` inputs (features)
"""
function neuron(n::Int,activation=tanh)
    w=value.(rand(n).*2 .-1)
    b=value(rand()*2-1)
    Neuron(w,b,activation)
end

# make a neuron a callable object
function (n::Neuron)(x)
    if length(x) != length(n.w)
        error("In calling n(x), expected $(length(n.w)) inputs to neuron, got $(length(x)):\n\tx = $(x)")
    end
    raw = sum(n.w .* x)+ n.b
    isnothing(n.activation) && return raw 
    return n.activation(raw)
end

mutable struct Layer{T<:AbstractFloat,F<:Union{Function,Nothing}}
    neurons::Vector{Neuron{T,F}}
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
function layer(n_in::Int,n_out::Int,activation=tanh)
    Layer([neuron(n_in,activation) for _ in 1:n_out],n_in,n_out)
end

# make layer callable
function (l::Layer)(x)
    out = [n(x) for n in l.neurons]
    length(out) == 1 && return out[1]
    return out
end

mutable struct MLP
    layers::Vector{Layer}
end

"""
    mlp(n_in,n_outs)
Construct a multi layer perceptron. `n_in` is the number of inputs to the MLP
and `n_outs` is a list of length `m` where `m` is the number of layers in the
MLP and `n_outs[i]` is the number of neurons in the `ith` layer.
"""
function mlp(n_in,n_outs,act=relu)
    n_vec = [n_in;n_outs]
    MLP([i!= length(n_outs) ? layer(n_vec[i],n_vec[i+1],act) : layer(n_vec[i],n_vec[i+1],nothing) for i in 1:length(n_outs)])
end

function (m::MLP)(x)
    for l in m.layers
        x = l(x)
    end
    return x
end

function parameters(n::Neuron)
    return [n.w... n.b]
end

function parameters(l::Layer)
    return collect(Iterators.flatten(parameters.(l.neurons)))
end

function parameters(m::MLP)
    return collect(Iterators.flatten(parameters.(m.layers)))
end


# --------------- loss and regularization --------------- #


function loss(m::MLP,X,y,type="l2")
    y_m = m.(X)
    if type=="l2"
        l = sum((y.-y_m).^2)
    elseif type=="l1"
        l = sum(abs(y.-y_m))
    elseif type=="hinge"
        l = sum(relu.(1.0 .+ -y.*y_m))/convert(eltype(y),length(y))
    else
        error("Unexpected loss type $type")
    end
    return l   

end

function regularization(m::MLP;alpha=0.0001,type="l2")
    p = parameters(m)
    if type=="l2"
        reg_l = alpha*sum(p.^2)
    end
    return reg_l
end


function objective(m::MLP,X,y,loss_type="hinge",reg_type="l2",show_accuracy=true)
    m_loss = loss(m,X,y,loss_type)
    reg_penalty = regularization(m,type=reg_type)
    total_loss =  m_loss + reg_penalty

    if show_accuracy
        a = accuracy(m,X,y)
        return (total_loss,a)
    end
    
    return total_loss

end

# --------------- optimizer --------------- #

"""
fit(m,X,y,...)

simple gradient descent with dynamic learning rate.
"""
function fit(m,X,y;loss_type="hinge",reg_type="l2",n_iter=100)
    
    for i in 0:n_iter-1

        # evaluate forward pass with current params
        total_loss,acc = objective(m,X,y,loss_type,reg_type)
        
        # reset then compute grads via backprop
        zerograd(m) # zero the model, don't need to zero from total loss since this is newly created
        backward(total_loss)
        
        #dynamic learning rate
        learning_rate = 1.0 - 0.9*i/100 

        # take a step
        for p in parameters(m)
            p.data -= learning_rate * p.grad
        end

        println("Step: $i\tLoss: $(round(total_loss.data,sigdigits=2))\tAccuracy: $(round(acc,sigdigits=2))")
    end
    
end

function zerograd(x::Union{Neuron,Layer,MLP,Neuron})
    for p in parameters(x)
        p.grad = 0
    end
end

# --------------- metrics --------------- #

function accuracy(m,X,y)
    return sum(sign.(getfield.(m.(X),:data)).==y)/length(y)
end