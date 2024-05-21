import { NeuralNetwork } from '../lib/index.js';
import mnist from 'mnist';

const TRAINING_ITERATIONS = 1000;
const LEARNING_RATE = 0.25;

const network = new NeuralNetwork([784, 10, 10]);
const { training, test } = mnist.set(900, 100);

// Train
for (let i = 0; i < TRAINING_ITERATIONS; i++) {
    console.log('Training iteration', i);
    network.train(training, LEARNING_RATE);
}

// Test
for (const { input, output } of test) {
    const result = network.predict(input);
    console.log('Test: input', input, 'expected output', output, 'actual output', result);
}