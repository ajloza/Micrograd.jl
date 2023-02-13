module Micrograd

# Write your package code here.
export Value

mutable struct Value{T<:AbstractFloat}
    data::T
    grad::T
    prev::Tuple
    op::Char
    backward::Function

end

function Value(d::T) where T
    Value(d,zero(T),(),' ',()->nothing)
end
function Value(d::T,c::Tuple) where T
    Value(d,zero(T),c,' ',()->nothing)
end
function Value(d::T,c::Tuple,o::Char) where T
    Value(d,zero(T),c,o,()->nothing)
end
function Value(d::T,g::T,c::Tuple,o::Char) where T
    Value(d,g,c,o,()->nothing)
end

Base.:+(x::Value{T},y::Value{T}) where T = Value(x.data+y.data,one(T),(x,y),'+')
Base.:-(x::Value{T},y::Value{T}) where T = Value(x.data-y.data,one(T),(x,y),'-')
Base.:*(x::Value{T},y::Value{T}) where T = Value(x.data*y.data,one(T),(x,y),'*')
Base.:/(x::Value{T},y::Value{T}) where T = Value(x.data/y.data,one(T),(x,y),'/')
Base.:^(x::Value{T},y::Value{T}) where T = Value(x.data^y.data,one(T),(x,y),'^')

end