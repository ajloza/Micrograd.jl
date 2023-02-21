# Micrograd

An implementation of Andrej Karpathy's Micrograd https://github.com/karpathy/micrograd in Julia.
 - reverse mode autodiff
 - basic NN
 - No dependencies.

Why? With such a clear guide in python, switching languages seemed closer to creating something to aid my understanding.

>"What I cannot create, I do not understand"
\- RPF

This is hopefully useful for education purposes. Any bugs were likely introduced in the porting from Andre

## Notes

1. A key component of Micrograd is the visualization of the computation graph. I could not find a simple clean DAG drawing framework that worked out of the box. Included is a bare-bones polytree drawer for ther terminal

## Contributing

