interface Neuron {
    /** Input connections from neurons in previous layer. */
    inputs: Connection[];
    /** Output connections to neurons in next layer. */
    outputs: Connection[];
    /** Neuron activation. */
    activation: number;
    /** Neuron bias. */
    bias: number;
    /** Temporary value used to compute cost gradients later. */
    delta: number;
    /** Sum of gradients of cost function with respect to the bias. */
    gradient: number;
}

interface Connection {
    /** Source neuron. */
    from: Neuron;
    /** Target neuron. */
    to: Neuron;
    /** Connection weight. */
    weight: number;
    /** Sum of gradients of cost function with respect to the weight. */
    gradient: number;
}

interface Layer {
    /** List of neurons in this layer. */
    neurons: Neuron[];
}

export class NeuralNetwork {
    protected readonly layers: Layer[];

    constructor(layerSizes: number[]) {
        this.layers = [];
        for (let i = 0, len = layerSizes.length; i < len; i++) {
            const layer: Layer = { neurons: [] };
            for (let n = 0, len = layerSizes[i]; n < len; n++) {
                const neuron: Neuron = {
                    inputs: [],
                    outputs: [],
                    activation: 0.0,
                    bias: Math.random() - 0.5,
                    delta: 0.0,
                    gradient: 0.0
                };
                if (i > 0) {
                    for (const prev of this.layers[i - 1].neurons) {
                        const connection: Connection = {
                            from: prev,
                            to: neuron,
                            weight: Math.random() - 0.5,
                            gradient: 0.0
                        };
                        prev.outputs.push(connection);
                        neuron.inputs.push(connection)
                    }
                }
                layer.neurons.push(neuron);
            }
            this.layers.push(layer);
        }
    }

    train(trainingSet: { input: number[]; output: number[]; }[], learningRate: number): void {
        this.clearGradients();
        for (const { input, output } of trainingSet) {
            this.updateActivations(input); // Set neuron activations
            this.calculateDeltas(output); // Calculate deltas using backpropagation
            this.updateGradients(); // Update cost gradients
        }
        this.updateWeightsAndBiases(trainingSet.length, learningRate);
    }

    predict(input: number[]): number[] {
        this.updateActivations(input);
        const outputLayer = this.layers[this.layers.length - 1];
        return outputLayer.neurons.map(neuron => neuron.activation);
    }

    protected updateActivations(input: number[]): void {
        // Set input neuron activations
        const inputLayerNeurons = this.layers[0].neurons;
        for (let n = 0, len = inputLayerNeurons.length; n < len; n++) {
            inputLayerNeurons[n].activation = input[n];
        }
        // Propagate forward
        for (let l = 1, len = this.layers.length; l < len; l++) {
            for (const neuron of this.layers[l].neurons) {
                let v = neuron.bias;
                for (const input of neuron.inputs) {
                    v += input.weight * input.from.activation;
                }
                neuron.activation = 1.0 / (1.0 + Math.exp(-v));
            }
        }
    }

    protected calculateDeltas(target: number[]): void {
        // Calculate deltas for output layer neurons
        const outputLayerNeurons = this.layers[this.layers.length - 1].neurons;
        for (let n = 0, len = outputLayerNeurons.length; n < len; n++) {
            const neuron = outputLayerNeurons[n];
            neuron.delta = (neuron.activation - target[n]) * neuron.activation * (1.0 - neuron.activation);
        }
        // Calculate deltas for previous layers using backpropagation
        for (let l = this.layers.length - 2; l >= 0; l--) {
            for (const neuron of this.layers[l].neurons) {
                let v = 0;
                for (const output of neuron.outputs) {
                    v += output.weight * output.to.delta;
                }
                neuron.delta = v * neuron.activation * (1.0 - neuron.activation);
            }
        }
    }

    protected updateGradients(): void {
        for (const layer of this.layers) {
            for (const neuron of layer.neurons) {
                for (const input of neuron.inputs) {
                    input.gradient += neuron.delta * input.from.activation;
                }
                neuron.gradient += neuron.delta;
            }
        }
    }

    protected clearGradients(): void {
        for (const layer of this.layers) {
            for (const neuron of layer.neurons) {
                for (const connection of neuron.inputs) {
                    connection.gradient = 0.0;
                }
                neuron.gradient = 0.0;
            }
        }
    }

    protected updateWeightsAndBiases(count: number, learningRate: number): void {
        for (const layer of this.layers) {
            for (const neuron of layer.neurons) {
                for (const connection of neuron.inputs) {
                    connection.weight -= (connection.gradient / count) * learningRate;
                }
                neuron.bias -= (neuron.gradient / count) * learningRate;
            }
        }
    }
}
