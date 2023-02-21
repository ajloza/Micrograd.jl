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

nodes,depth = buildgraph(o)

printgraph(nodes,depth)
