
# --------------- Value types --------------- #
 
# make a callable concrete type for the backward function.
struct Backward{F<:Function} f::F end
(b::Backward)() = b.f()
const noback = Backward(()->nothing)

# main type to handle relations and operations for backprop
mutable struct Value{T<:AbstractFloat}
    data::T
    grad::T
    prev::Tuple
    op::String
    bw::Backward
end

value(d::T,g::T=zero(T),c=(),o="",bw=noback) where T = Value(d,g,c,o,bw)


# --------------- ops --------------- #

function Base.:+(x::Value{T},y::Value{T}) where T 
    out = value(x.data+y.data,zero(T),(x,y),"+")
    function bwf()
        x.grad += out.grad  # dL/dx = dout/dx * dL/dout = 1.0 (local) * out.grad (passed) = out.grad
        y.grad += out.grad
        return nothing
    end
    out.bw = Backward(bwf)
    return out
end
Base.:+(x::Value{T},y::T) where T = x+value(y)
Base.:+(y::T,x::Value{T}) where T = x+value(y)

# define subtraction in terms of unary minus and plus
Base.:-(x::Value{T},y::Value{T}) where T = x+(-y)
Base.:-(x::Value{T},y::T) where T = x-value(y)
Base.:-(y::T,x::Value{T}) where T = value(y)-x
Base.:-(x::Value{T}) where T = x*-1.0

function Base.:*(x::Value{T},y::Value{T}) where T 
    out = value(x.data*y.data,zero(T),(x,y),"*")
    function bwf()
        x.grad += y.data*out.grad
        y.grad += x.data*out.grad
    end
    out.bw = Backward(bwf)
    return out
end
Base.:*(x::Value{T},y::T) where T = x*value(y)
Base.:*(y::T,x::Value{T}) where T = x*value(y)

function Base.:^(x::Value{T},y::Union{I,F}) where {T,I<:Integer, F<:AbstractFloat}
    out = value(x.data^y,zero(T),(x,),"^")
    function bwf()
        x.grad += (y*x.data^(y-1))*out.grad
    end
    out.bw = Backward(bwf)
    return out
end
function Base.inv(x::Value{T}) where T
    out = value(x.data^-1,zero(T),(x,),"^")
    function bwf()
        x.grad += -1*x.data^(-2)*out.grad
    end
    out.bw = Backward(bwf)
    return out
end

Base.:/(x::Value{T},y::Value{T}) where T = x*y^-1
Base.:/(x::T,y::Value{T}) where T = value(x)*y^-1
Base.:/(x::Value{T},y::T) where T = x*value(y)^-1

function Base.exp(x::Value{T}) where T
    out_data = exp(x.data)
    out = value(out_data,zero(T),(x,),"exp")
    function bwf()
        x.grad += out_data*out.grad
    end
    out.bw = Backward(bwf)
    return out
end

# some activations
function Base.tanh(x::Value{T}) where T
    out_data = tanh(x.data)
    out = value(out_data,zero(T),(x,),"tanh")
    function bwf()
        x.grad += (1-out_data^2)*out.grad
    end
    out.bw = Backward(bwf)
    return out
end

function relu(x::Value{T}) where T
    out_data = x.data > 0.0 ? x.data : 0.0
    out = value(out_data,zero(T),(x,),"relu")
    function bwf()
        x.grad += x.data > 0.0 ? out.grad : 0.0   # dL/dx = dout/dx (1.0 or 0.0) *  dL/dout (out.grad)
    end
    out.bw = Backward(bwf)
    return out
end

Base.length(x::Value) = 1
Base.iterate(x::Value,y) = nothing
Base.iterate(x::Value) = (x,1)


# --------------- computational graph and backprop --------------- #

function backward(x::Value)
    nodes,_ = buildgraph(x)
    x.grad = 1.0
    for n in reverse(nodes)
        n.bw()
    end
    return nothing
end

function zerograd(x::Value)
    nodes,_ = buildgraph(x)
    for n in nodes
        n.grad = 0
    end
    return nothing
end



function buildgraph(x::Value)

    topo = Vector{Value}()
    topo_depth = Vector{Int}()
    visited = Vector{Value}()
    depth = 1
    buildtopo(x,topo,visited,topo_depth,depth)
    return (topo,topo_depth)

end

function buildtopo(x,topo,visited,topo_depth,depth)
    if !(x in visited)
        push!(visited,x)
        for p in x.prev
            buildtopo(p,topo,visited,topo_depth,depth+1)
        end
        push!(topo,x)
        push!(topo_depth,depth)
    end
end

