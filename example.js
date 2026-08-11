import {Value} from './engine.js'
import {MultiLayerPerceptron} from './neuralnet.js'

const model = new MultiLayerPerceptron(1, [4, 1]);

const xs = [
[0],
[1],
[2],
[3],
[4],
[5]
];
const ys = [3,13,23,33,43,53];

// Hyperparameters
const learningRate = 0.05;
const epochs = 1000;

console.log(`Training over ${epochs} epochs with learning rate ${learningRate}`);

// Training Loop
for (let epoch = 0; epoch < epochs; epoch++) {

    // ---- Zero gradients for all parameters ----
    // model.parameters().forEach(p => p.grad = 0);
    model.zeroGrad();


    // ---- Forward Pass ----
    const ypred = xs.map(x => model.activate(x, 'tanh', 'linear'));

    let loss = ypred.map((pred, i) => pred.sub(ys[i]).pow(2)).reduce((acc, curr) => acc.add(curr), new Value(0));
    
    loss = loss.div(xs.length); // Average loss


    // for (let i = 0; i < ys.length; i++) {
    //     const diff = ypred[i].sub(ys[i]);
    //     loss = loss.add(diff.pow(2));
    // }

    // ---- Backward Pass ----
    loss.backward();

    // ---- SGD Update ----
    model.parameters().forEach(p => p.data -= learningRate * p.grad);


    // ---- Logging ----
    if (epoch % 10 === 0 || epoch === epochs - 1) {
        console.log(`Epoch ${epoch.toString()} | Loss: ${loss.data.toFixed(6)}`);
    }
}


// Verification
console.log("Prediction vs Target");
xs.forEach((x, i) => {
    const pred = model.activate(x, 'tanh', 'linear');

    console.log(`Input: ${i + 1} | Predicted: ${pred.data.toFixed(4)} | Target: ${ys[i]}`);
})