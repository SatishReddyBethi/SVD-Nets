# SVD-Nets
This repo contains my experiments for training neural networks using SVD instead of gradient descent.

Initial Results:
You can definitely replace gradient descent with SVD for training a neural network

Pros:
- No need to use cost function
- No need to calculate gradients
- We can build optimal model architectures (models with optimal number of layers and optimal number of nodes for each layer) by keeping an eye on the singular values
- Testing and Inference process is the same as neural nets with gradient descent
- Always converges to global minimum

Cons:
- Takes too long to train when compared to gradient descent: ~10x times* (* Trained only on mnist dataset)
- Takes more memory to train as we have to also save principal vectors and values

Things to keep in mind:
- Since this prototype is written from scratch, it's not completely fair to compare with neural networks from libraries which have been optimized a lot more over the last decade.
