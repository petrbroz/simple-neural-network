# simple-neural-network

Stupid simple implementation of a fully-connected artificial neural network for learning purposes. Based on various online resources, incl. https://www.youtube.com/watch?v=sIX_9n-1UbM.

> Note however that there seems to be a small error in the YouTube video. Around the 3:58 time mark a derivative of the cost function $(\hat{a_1} - a_1^{(4)})^2$ is said to be $2(\hat{a_1} - a_1^{(4)})$ while it should be $2(a_1^{(4)} - \hat{a_1})$.

## Usage

```js
// Create neural network with specific number of layers and neurons,
// in this case with an input layer of 2 neurons, one hidden layer of 2 neurons, and an output layer of 1 neuron
const network = new NeuralNetwork([2, 2, 1]);

// Repeatedly train the network with 0.25 learning rate
const trainingData = [
    { input: [0.0, 0.0], output: [1.0] },
    { input: [0.0, 1.0], output: [0.0] },
    { input: [1.0, 0.0], output: [0.0] },
    { input: [1.0, 1.0], output: [1.0] }
];
for (let i = 0; i < 1000; i++) {
    network.train(trainingData, 0.25);
}

// Test
for (let y = 0.0; y <= 1.0; y += 0.1) {
    for (let x = 0.0; x <= 1.0; x += 0.1) {
        console.log(network.predict([x, y]));
    }
}
```