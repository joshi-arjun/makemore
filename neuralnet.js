import { Value } from './engine.js';


/**
 * Neuron class represents a single neuron in a neural network
 */
export class Neuron {
    #nin;

    /**
     * 
     * @param {number} nin - Number of input connections to the neuron 
     * @param {string|number} label - Debug Label for the neuron
     * @param {number} layer - Index of layer containing this neuron for debugging
     */
    constructor (nin, label, layer) {
        this.#nin = nin;
        this.layer = layer;
        this.label = label;

        // Random weights and biases from range [-1,1]
        this.w = Array.from({ length: nin }, () => new Value(Math.random() * 2 -1));
        this.b = new Value (Math.random() * 2 - 1);
        
    }

    /**
     * 
     * @param {Value[]|number[]} x - Array of inputs 
     * @param {'tanh'|'relu'|'linear'} activationFn - Activation function to use for this neuron 
     * @returns {Value} Accumulated activation
     */
    activate (x, activationFn) {
        // Ensure inputs are wrapped as Value instances
        const inputs = x.map(val => Value.from(val));

        // Start accumulator with bias
        let activation = this.b;

        // Compute weighted sum
        for (let i = 0; i < this.#nin; i++) {
            activation = activation.add(this.w[i].mult(inputs[i]));
        }

        switch (activationFn) {
            case 'relu':
                return activation.relu();
            case 'linear':
                return activation;
            case 'tanh':
            default:
                return activation.tanh();
        }
    }

    /** @returns {Value[]} Flattened list of mutable training paramaters */
    parameters () {
        return [...this.w, this.b];
    }
}


/**
 * Layer containing multiple neurons connected to the same input vector
 * Each neuron in the layer has its own weights and bias
 * The number of outputs determines the number of neurons in this layer
 */
export class Layer {
    
    /**
     * 
     * @param {number} nin - Number of inputs per neuron
     * @param {number} nout - Number of neurons in the layer
     * @param {number} layerIndex - Index of this layer in the network for debugging 
     */
    constructor (nin, nout, layerIndex) {
        this.layerIndex = layerIndex;
        this.neurons = Array.from(
            {length: nout},
            (_, i) => new Neuron(nin, i + 1, layerIndex)
        );
    }

    /**
     * 
     * @param {Value[]|number[]} x - Input vector to the layer 
     * @param {'tanh'|'relu'|'linear'} [activationFn='tanh']
     * @returns {Value[]} Array of activations from each neuron in the layer
     */
    activate (x, activationFn = 'tanh') {
        return this.neurons.map(neuron => neuron.activate(x, activationFn));
    }

    /** @returns {Value[]} Flattened list of mutable training paramaters */
    parameters () {
        return this.neurons.flatMap(neuron => neuron.parameters());
    }
}

/**
 * Fully connected Multi-Layer Perceptron (MLP) Architechture
 */
export class MultiLayerPerceptron {
    /**
     * 
     * @param {number} nin - Number of network inputs 
     * @param {number[]} nouts - Array of numbers representing the number of neurons in each layer
     */
    constructor (nin, nouts) {
        const layerSizes = [nin, ...nouts];

        this.layers = nouts.map((nout, i) => new Layer(layerSizes[i], nout, i + 1));

    }
    /**
     * Forward pass through the network, activating each layer in sequence
     * Uses tanh for hidden layers and linear activation for the output layer
     * @param {Value[]|number[]} x - Input Vector
     * @param {'tanh'|'relu'} [hiddenActivationFn='tanh']
     * @param {'tanh'|'relu'|'linear'} [outputActivationFn='linear']
     * @returns {Value|Value[]} Output scalar or vector.
     */
    activate (x, hiddenActivationFn = 'tanh', outputActivationFn = 'linear') {
        for (let i = 0; i < this.layers.length; i++) {
            const isLastLayer = (i === this.layers.length - 1);
            const actFn = isLastLayer ? outputActivationFn : hiddenActivationFn;

            x = this.layers[i].activate(x, actFn);
        }
        // Unwraps single scalar outputs for easier logging and loss computation.
        return (x.length == 1) ? x[0] : x;
    }
    /** @returns {Value[]} Flattened list of mutable training parameters */
    parameters () {
        return this.layers.flatMap(layer => layer.parameters());
    }

    /**Resets all Paramater gradients to zero before the backward pass */
    zeroGrad() {
        for (const p of this.parameters()) {
            p.grad = 0;
        }
    }
}
