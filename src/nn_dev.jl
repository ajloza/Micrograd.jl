using Micrograd
## make neuron
n = neuron(3,relu)
x = [1.0,2.0,1.0]
o = n(x)

m = mlp(3,[1])
X = [x]
y = 1.0
m.(X)
fit(m,X,y)

try
    n([1.0,2.0])
catch e
    println(e)
    println("\n ^ correctly throws error")
end

# make a layer with 3 inputs and 10 outputs
l = layer(3,10)
l(x)

# make a mlp with 3 inputs and 3 layers of size 4, 4, then 1
m = mlp(3,[4,4,1])
m.layers[1].neurons[1].w[1]

# don't try this is wasn't made for it
o = m(x)
nodes,depth = buildgraph(o)
printgraph(nodes,depth)

# set up the binary classifier from Micrograd
X = [
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0],
]

y = [1.0,-1.0,-1.0,-1.0]

y_m = m.(X)

#checks
loss(m,X,y,"hinge")
regularization(m)
accuracy(m,X,y)
objective(m,X,y)

fit(m,X,y)

parameters(m)
# check that grads are zero

backward(m_loss)

# see the new grad
m.layers[1].neurons[1].w[1]

parameters(n)
parameters(l)
p = parameters(m)

X,y = getmoons()

m = mlp(2, [16, 16, 1])
l,_ = objective(m,X,y)
backward(l)

loss(m,X,y)

fit(m,X,y)