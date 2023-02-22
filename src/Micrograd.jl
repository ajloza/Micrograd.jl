

module Micrograd

using StaticArrays

export Value, value, Backward
export Neuron, neuron, Layer, layer, MLP, mlp
export buildgraph, buildtopo
export backward, zerograd
export printgraph

include("autodiff.jl")
include("printing.jl")
include("nn.jl")



end