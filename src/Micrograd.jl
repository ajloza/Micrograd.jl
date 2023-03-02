

module Micrograd

export Value, value, Backward
export Neuron, neuron, Layer, layer, MLP, mlp, loss, parameters, relu
export buildgraph, buildtopo
export backward, zerograd
export printgraph
export objective, regularization, accuracy, fit
export getmoons

include("autodiff.jl")
include("nn.jl")
include("printing.jl")
include("datasets.jl")



end