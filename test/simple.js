import { stdout } from 'node:process';
import { NeuralNetwork } from '../lib/index.js';

const TRAINING_ITERATIONS = 10_000;
const LEARNING_RATE = 0.25;
const TRAINING_DATA = [
    { input: [0.0, 0.0], output: [0.0] },
    { input: [0.0, 1.0], output: [0.5] },
    { input: [1.0, 0.0], output: [0.5] },
    { input: [1.0, 1.0], output: [1.0] }
];

// Creating neural network with an input layer with 2 neurons, one hidden layer with 2 neurons, and an output layer with 1 neuron
let network = new NeuralNetwork([2, 2, 1]);

// Repeatedly train the network with the same training data
for (let i = 0; i < TRAINING_ITERATIONS; i++) {
    console.log('Training iteration', i);
    network.train(TRAINING_DATA, LEARNING_RATE);
}

// Test
for (let y = 0.0; y <= 1.0; y += 0.1) {
    for (let x = 0.0; x <= 1.0; x += 0.1) {
        const result = network.predict([x, y]);
        stdout.write(result[0].toFixed(2) + ' ');
    }
    stdout.write('\n');
}