# Micrograd

An implementation of Andrej Karpathy's Micrograd https://github.com/karpathy/micrograd in Julia.
 - reverse mode autodiff
 - basic NN
 - No dependencies.

Why? With such a clear guide in python, switching languages seemed closer to creating something to aid my understanding.

>"What I cannot create, I do not understand"
\- RPF

This is hopefully useful for education purposes. Any bugs were likely introduced in the porting from Andrej's implementation.

## Notes

1. A key component of Micrograd is the visualization of the computation graph. I could not find a simple clean DAG drawing framework that worked out of the box and maintining the repo as dependency free was a priority. Included is a bare-bones polytree drawer for the terminal.

## Contributing

