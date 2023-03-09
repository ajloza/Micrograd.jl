# Micrograd

An implementation of Andrej Karpathy's Micrograd https://github.com/karpathy/micrograd in Julia. 
 - reverse mode autodiff
 - basic NN
 - No dependencies.

Why? With such a clear guide in python, switching languages seemed closer to creating something.
>"What I cannot create, I do not understand"
\- RPF

This is hopefully useful for education purposes. Any bugs were likely introduced in the porting from Andrej's implementation.

## Notes

1. This initially done for myself and for a local teaching tool, but I'm placing it here in case others find it useful.
2. A key component of learning from Micrograd is the visualization of the computation graph. I could not find a simple clean DAG drawing framework that worked out of the box and also wanted to keep the code dependency free. Included is a bare-bones polytree drawer for the terminal. It does not show shared parents.
3. This was done in Julia for the above reasons, not for performance. I have not really optimized any step beyond what came out when coding it (I'm not sure I even want to see the allocations or a flamegraph of `fit()` considering it rebuilds the forward graph each time). At the same time, it's not meant to be anti-performant, so if you see anything that improves performance but (a) doesn't change code clarity (b) doesn't stray too far from the framework of Micrograd, please file a PR. It also may not be the best "Julian" code.

## Contributing

- If you'd like to contribute, I'd welcome it! I'm trying to keep things as simple as possible.
- If you notice any bugs, performance enhancements, improved coding stye, or any other issues, please submit a PR!

