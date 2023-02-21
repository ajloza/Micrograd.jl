using Micrograd

a = value(1.0)
b = value(2.0)

c = a+b
c.grad = 1.0
c.data
c.bw.f()
c
c.bw()

a
c.grad = 2

backward(c)
a
b

tanh(a)