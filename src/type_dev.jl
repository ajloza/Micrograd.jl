using Micrograd

## simple addition
a = value(1.0)
b = value(2.0)
c = a+b
d = tanh(c)

nodes,depth = buildgraph(d)
printgraph(nodes,depth)

# pre-backward all grads are 0.0
a.grad

# set the gradient of d and call the bw function
d.grad = 1.0
d.bw()
c.bw()
printgraph(nodes,depth)

## single neuron example, replicating micrograd example
x1 = value(2.0)
w1 = value(-3.0)
x2 = value(0.0)
w2 = value(1.0)
b = value(6.8813735870195432)

x1w1 = x1*w1
x2w2 = x2*w2

x1w1x2w2 = x1w1+x2w2
n = x1w1x2w2+b
o = tanh(n)

backward(o)

nodes,depth = buildgraph(o)

printgraph(nodes,depth)

## double call
a = value(2.0)
b = value(1.0)
c = a-b

nodes,depth = buildgraph(c)
printgraph(nodes,depth)

## make neuron
n = neuron(3)
x = [1.0,2.0,1.0]
o = n(x)

try
    n([1.0,2.0])
catch e
    println(e)
    println("correctly throws error")
end

# make a layer with 3 inputs and 10 outputs
l = layer(3,10)
l(x)

# make a mlp with 3 inputs and 3 layers of size 4, 4, then 1
m = mlp(3,[4,4,1])

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

y = [1.0,-1.0,-1.0,1.0]

y_pred = m.(X)
m_loss = loss(y,y_pred,"l2")

# check that grads are zero
m.layers[1].neurons[1].w[1]

backward(m_loss)

# see the new grad
m.layers[1].neurons[1].w[1]

parameters(n)
parameters(l)
p = parameters(m)