/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import * as tfd from '@tensorflow/tfjs-data';

import {ControllerDataset} from './controller_dataset';
import * as ui from './ui';

// The number of classes we want to predict. In this example, we will be
// predicting 4 classes for up, down, left, and right.
const NUM_CLASSES = 4;

// A webcam iterator that generates Tensors from the images from the webcam.
let webcam;

// The dataset object where we will store activations.
const controllerDataset = new ControllerDataset(NUM_CLASSES);

let truncatedMobileNet;
let model;
let layer_pre_softmax;
let model_pre_softmax;

async function loadTruncatedMobileNet() {
    const mobilenet = await tf.loadLayersModel(
        'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

    const layer = mobilenet.getLayer('conv_pw_13_relu');
    return tf.model({inputs: mobilenet.inputs, outputs: layer.output});

}

ui.setExampleHandler(async label => {
    let img = await getImage();

    controllerDataset.addExample(truncatedMobileNet.predict(img), label);

    // Draw the preview thumbnail.
    ui.drawThumb(img, label);
    img.dispose();
});

/**
 * Sets up and trains the classifier.
 */
async function train() {
    if (controllerDataset.xs == null) {
        throw new Error('Add some examples before training!');
    }
    //model define
    const input = tf.layers.input({shape: [7, 7, 256]});
    const flatten = tf.layers.flatten().apply(input);
    const dense1 = tf.layers.dense({
        units: ui.getDenseUnits(),
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
    }).apply(flatten);
    const dense2 = tf.layers.dense({
        name:"pre_softmax",
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
    }).apply(dense1);
    const softmax = tf.layers.softmax({}).apply(dense2);
    model = tf.model({inputs: input, outputs: softmax});

    const optimizer = tf.train.adam(ui.getLearningRate());

    model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

    const batchSize =
        Math.floor(controllerDataset.xs.shape[0] * ui.getBatchSizeFraction());
    if (!(batchSize > 0)) {
        throw new Error(
            `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
    }

    // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
    model.fit(controllerDataset.xs, controllerDataset.ys, {
        batchSize,
        epochs: ui.getEpochs(),
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                ui.trainStatus('Loss: ' + logs.loss.toFixed(5));
            }
        }
    });

    layer_pre_softmax = model.getLayer("pre_softmax");
    model_pre_softmax = tf.model({inputs: model.inputs, outputs: layer_pre_softmax.output});
    await model_pre_softmax.save('indexeddb://model');
}

let isPredicting = false;

async function predict() {
    while (isPredicting) {
        // Capture the frame from the webcam
        const img = await getImage();

        const embeddings = truncatedMobileNet.predict(img);

        const predictions = model_pre_softmax.predict(embeddings);
        console.log(predictions.as1D().data());

        const predictedClass = predictions.as1D().argMax();
        const classId = (await predictedClass.data())[0];
        img.dispose();

        ui.predictClass(classId);
        await tf.nextFrame();
    }
}

/**
 * Captures a frame from the webcam and normalizes it between -1 and 1.
 * Returns a batched image (1-element batch) of shape [1, w, h, c].
 */
async function getImage() {
    const img = await webcam.capture();
    const processedImg =
        tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
    img.dispose();
    return processedImg;
}

document.getElementById('train').addEventListener('click', async () => {
    ui.trainStatus('Training...');
    await tf.nextFrame();
    await tf.nextFrame();
    isPredicting = false;
    train();
});
document.getElementById('predict').addEventListener('click', () => {
    ui.startPacman();
    isPredicting = true;
    predict();
});

async function init() {
    try {
        webcam = await tfd.webcam(document.getElementById('webcam'));
    } catch (e) {
        console.log(e);
        document.getElementById('no-webcam').style.display = 'block';
    }
    truncatedMobileNet = await loadTruncatedMobileNet();

    ui.init();

    // Warm up the model. This uploads weights to the GPU and compiles the WebGL
    // programs so the first time we collect data from the webcam it will be
    // quick.
    const screenShot = await webcam.capture();
    truncatedMobileNet.predict(screenShot.expandDims(0));
    screenShot.dispose();
}

// Initialize the application.
init();
