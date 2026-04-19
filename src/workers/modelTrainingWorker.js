import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Worker de treinamento de modelo inicializado');

let _globalCtx = {};

let _model = null;

const normalizeRule = (value, min, max) => ((value - min) / (max - min) || 1);

const oneHotWeighted = (index, lenght, weight) => tf.oneHot(index, lenght).cast('float32').mul(weight);

const weights = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1
};

function makeContext(catalog, users){
    const ages = users.map(u => u.age);
    const prices = catalog.map(p => p.price);

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);
    
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    const colors = [...new Set(catalog.map(p => p.color))];
    const categories = [...new Set(catalog.map(p => p.category))];

    const colorIndexes = Object.fromEntries(
        colors.map(
            (color, index) => [color, index]
        )
    );

    const categoriesIndexes = Object.fromEntries(
        categories.map(
            (category, index) => [category, index]
        )
    );

    const midAge = (minAge + maxAge) / 2;
    const ageSums = {};
    const ageCounts = {};

    users.forEach(u => {
        u.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + u.age;
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1;
        })
    });

    // determino a média de idade de pessoas que compraram determinado produto
    const productAgeAvgNorm = Object.fromEntries(
        catalog.map(
            product => {
                const avg = ageCounts[product.name] ? ageSums[product.name] / ageCounts[product.name] : midAge
                return [product.name, normalizeRule(avg, minAge, maxAge)]
            }
        )
    );

    /**
     * dimentions do retorno corresponde ao nº de categorias + nº de cores + 2 (preço e idade)
     * basicamente será a dimensão do objeto de entrada de treinamento do modelo
     */
    return {
        catalog, 
        users, 
        colorIndexes, 
        categoriesIndexes, 
        minAge, 
        maxAge, 
        minPrice, 
        maxPrice, 
        qtdCategories: categories.length, 
        qtdColors: colors.length, 
        dimentions: 2 + categories.length + colors.length,
        productAgeAvgNorm
    }
}

function encodeProduct(product, context) {
    const price = tf.tensor1d([
        normalizeRule(product.price, context.minPrice, context.maxPrice) * weights.price
    ]);

    const age = tf.tensor1d([
        (context.productAgeAvgNorm[product.name] ?? 0.5) * weights.age
    ]);

    const category = oneHotWeighted(
        context.categoriesIndexes[product.category], context.qtdCategories, weights.category
    );

    const colors = oneHotWeighted(
        context.colorIndexes[product.color], context.qtdColors, weights.color
    );

    return tf.concat1d([price, age, category, colors]);
}

function encodeUser(user, context){
    if(user.purchases.length > 0){
        return tf.stack(
            user.purchases.map(p => encodeProduct(p, context))
        ).mean(0)
         .reshape([1, context.dimentions]);
    }

    return tf.concat1d(
        [
            tf.zeros([1]), // ignora o preço
            tf.tensor1d([
                normalizeRule(user.age, context.minAge, context.maxAge) * weights.age
            ]),
            tf.zeros([context.qtdCategories]), // ignora as categorias
            tf.zeros([context.qtdColors]) // ignora as cores
        ]
    ).reshape([1, context.dimentions]);
}

function createTrainingData(context){
    const inputs = [];  
    const labels = [];

    context.users.filter(user => user.purchases.length > 0).forEach(user => {
        const userVector = Array.from(encodeUser(user, context).dataSync());

        context.catalog.forEach(product => {
            const productVector = Array.from(encodeProduct(product, context).dataSync());

            const label = user.purchases.some(p => p.name === product.name) ? 1 : 0;

            inputs.push([
                ...userVector,
                ...productVector
            ]);

            labels.push(label);
        });
    });

    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimension: context.dimentions * 2
    };
}

async function configureNeuralNetAndTrain(trainData){
    const model = tf.sequential();

    model.add(
        tf.layers.dense({
            inputShape: [trainData.inputDimension],
            units: 128,
            activation: 'relu'
        })
    );

    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu'
        })
    );

    model.add(
        tf.layers.dense({
            units: 32,
            activation: 'relu'
        })
    );

    model.add(
        tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        })
    );

    model.compile(
        {
            optimizer: tf.train.adam(0.01),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        }
    );

    await model.fit(
        trainData.xs, 
        trainData.ys,
        {
            epochs: 100,
            batchSize: 32,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    postMessage({
                        type: workerEvents.trainingLog,
                        epoch: epoch,
                        loss: logs.loss,
                        accuracy: logs.acc
                    });
                }
            }
        }
    );

    return model;
}

async function trainModel({ users }) {
    console.log('Treinando modelo com os usuários:', users)
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

    const catalog = await (await fetch('/data/products.json')).json();

    const context = makeContext(catalog, users);

    context.productVectors = catalog.map(product => {
        return {
            name: product.name,
            meta: {...product},
            vector: encodeProduct(product, context).dataSync()
        }
    })

    _globalCtx = context;

    const trainData = createTrainingData(context);
    _model = await configureNeuralNetAndTrain(trainData);

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}

function recommend(user, ctx) {
    if(!_model){
        console.error('Modelo não treinado');
        return;
    }

    const context = ctx;

    const userTensor = encodeUser(user, ctx).dataSync();

    console.log('Recomendando para o usuário:', user)

    const inputs = context.productVectors.map(({vector})=> {
        return [
            ...userTensor,
            ...vector
        ];
    });

    const inputTensor = tf.tensor2d(inputs);

    const predictions = _model.predict(inputTensor);

    const scores = predictions.dataSync();

    const recommendations = context.productVector.map((item, index)=> {
        return {
            ...item.meta,
            name: item.name,
            score: scores[index]
        }
    });

    const sortedItens = recommendations.sort((a, b) => b.score - a.score);

    postMessage({
         type: workerEvents.recommend,
         user,
         recommendations: sortedItens
    });
}

const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
