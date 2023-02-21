module Micrograd

export Value, value, Backward

# make a concrete type for the backward function.
# if we use ::F where {F<:Function} in Value() we can't change the function
struct Backward{F<:Function} f::F end
(b::Backward)() = b.f()

# null backward fn
const noback = Backward(()->nothing)

# main type to handle relations and operations for reverse mode backprop
mutable struct Value{T<:AbstractFloat}
    data::T
    grad::T
    prev::Tuple
    op::String
    bw::Backward
end

# constructor with defaults
value(d::T,g::T=zero(T),c=(),o="",bw=noback) where T = Value(d,g,c,o,bw)

# operations
function Base.:+(x::Value{T},y::Value{T}) where T 
    out = value(x.data+y.data,zero(T),(x,y),"+")
    function bwf()
        x.grad += out.grad # from: 1.0 (local grad) * out.grad (passed grad)
        y.grad += out.grad
        return nothing
    end
    out.bw = Backward(bwf)
    return out
end

Base.:-(x::Value{T},y::Value{T}) where T = Value(x.data-y.data,zero(T),(x,y),"-")

function Base.:*(x::Value{T},y::Value{T}) where T 
    out = value(x.data*y.data,zero(T),(x,y),"*")
    function bwf()
        x.grad += y.data*out.grad
        y.grad += x.data*out.grad
    end
    out.bw = Backward(bwf)
    return out
end

Base.:/(x::Value{T},y::Value{T}) where T = Value(x.data/y.data,zero(T),(x,y),"/")

function Base.:^(x::Value{T},y::Union{I,F}) where {T,I<:AbstractInt, F<:AbstractFloat}
    out = value(x.data^y,zero(T),(x,),"^")
    function bwf()
        x.grad += (y*x.data^(y-1))*out.grad
    end
    out.bw = Backward(bwf)
    return out
end


function Base.tanh(x::Value{T}) where T
    out_data = tanh(x.data)
    out = value(out_data,zero(T),(x,),"tanh",Backward(bwf))
    function bwf()
        x.grad += (1-out_data^2)*out.grad
    end
    out.bw = Backward(bwf)
    return out
end

# some formatting
function prettyvalue(x::Value)
    op_str = x.op=="" ? "none" : x.op
    str = string(x.data," (grad: ",x.grad,", op: ",op_str,")")
    return str
end
Base.show(io::IO, x::Value) = print(io,"$(prettyvalue(x))")
Base.show(io::IO,m::MIME"text/plain", x::Value) = print(io,"$(prettyvalue(x))")

end