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

1. A key component of Micrograd is the visualization of the computation graph. I could not find a simple clean DAG drawing framework that worked out of the box and keeping the code dependency free was a priority. Included is a bare-bones polytree drawer for the terminal. It does not show shared parents.
2. This was done in Julia for the above reasons, not for performance. I have not optimized any step beyond what came out when coding it (I'm not sure I even want to see the allocations or a flamegraph of `fit()`). At the same time, it's not meant to be anti-performant, so if you see anything that improves performance but (a) doesn't change code clarity (b) doesn't stray too far from the framework of Micrograd, please file a PR

## Contributing

- This initially done for myself and for a local teaching tool, but I'm placing it here in case others find it useful
- If you'd like to contribute, I'd welcome it! I'm trying to keep things as simple as possible.
- If you notice any bugs, performance enhancements, or improved docs, please submit a PR

