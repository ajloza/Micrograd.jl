

module Micrograd

export Value, value, Backward
export Neuron, neuron, Layer, layer, MLP, mlp, loss, parameters
export buildgraph, buildtopo
export backward, zerograd
export printgraph

include("autodiff.jl")
include("nn.jl")
include("printing.jl")



end