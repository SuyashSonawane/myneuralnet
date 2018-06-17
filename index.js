let lrate=parseFloat( document.getElementById('lrate').value)
let itter =parseInt( document.getElementById('itterations').value)
let epoch_user =parseInt( document.getElementById('epoch').value)
let select_loss = tf.losses.meanSquaredError;
let select_opt = tf.train.sgd(lrate); 
let select_activation = 'sigmoid'
let i = 0;
var optimizer = document.getElementById("optimizer");
var loss = document.getElementById("loss_function");
function optchange()
{
    var optimizer = document.getElementById("optimizer");
    switch(optimizer.options[optimizer.selectedIndex].value){
        case 'SGD':
            select_opt=tf.train.sgd(lrate)
            break;
        case 'Adagrad':
            select_opt = tf.train.adagrad(lrate)
            break;
        case 'Adadelta':
            select_opt = tf.train.adadelta(lrate)
            break;
        case 'Adam':
            select_opt = tf.train.adam(lrate)
            break;
        case 'Adamax':
            select_opt = tf.train.adamax(lrate)
            break;
        case 'Rmsprop':
            select_opt = tf.train.rmsprop(lrate)
            break; 
        default:
            select_opt=tf.train.sgd(lrate);     


    }
    console.log('Optimizer change detected ')
    alert('Improper optimizer may yield wierd results , choose them carefully')

}

function losschange() {

    var loss = document.getElementById("loss_function");
    switch (loss.options[loss.selectedIndex].value) {
        case 'AbsoluteDifference':
            select_loss = tf.losses.absoluteDifference
            break;
        case 'ComputeweightedLoss':
            select_loss = tf.losses.computeWeightedLoss
            break;
        case 'CosineDistance':
            select_loss = tf.losses.cosineDistance
            break;
        case 'MeanSquaredError':
            select_loss = tf.losses.meanSquaredError
            break;
        case 'Logloss':
            select_loss = tf.losses.logLoss
            break;
        case 'SoftmaxCrossEntropy':
            select_loss = tf.losses.softmaxCrossEntropy
            break;
        default:
            select_loss = tf.losses.meanSquaredError


    }
    console.log('Loss function change detected ')
    alert('Improper loss function may yield wierd results , choose them carefully')
    
}

function actchange() {
    var activaion = document.getElementById("activation");
    select_activation = activaion.options[activaion.selectedIndex].value    
}


const model = tf.sequential()

const l2 = tf.layers.dense(
    {
        units:4,
        inputShape:[3],                            // input nodes
        activation:select_activation                        //configs for hidden layer
    });

model.add(l2)                                         // adding h layer to model

const l3 = tf.layers.dense(
    {
        units:1,
        activation:select_activation
    });                                                  // configs for output layer

model.add(l3)                                           // addding o layer to model


function main(){    
   
    if (i == 0) { alert('Pc users check console for more info about the process') ; i=1}
    

    const config =
        {
            optimizer: select_opt,                        //config  for model compilation
            loss: select_loss
        }

document.getElementById('txtarea').innerText = 'Running network please wait'
model.compile(config)
let lrate = parseFloat(document.getElementById('lrate').value)
let itter = parseInt(document.getElementById('itterations').value)
let epoch_user = parseInt(document.getElementById('epoch').value)


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                 Generating data :)

const x = tf.tensor2d([
    [0,1, 0],
    [1,0,1 ],
    [1, 1,1]
])

const y = tf.tensor2d([
    [1],
    [0],
    [1]
])

train();

    let model_info = 'The learning rate :: '
        + lrate
        + '\nItterations :: '
        + itter
        + '\nEpoch are :: '
        + epoch_user
        + '\nThe optimizer is :: '
        + optimizer.options[optimizer.selectedIndex].value
        + '\nThe loss function is :: '
        + loss.options[loss.selectedIndex].value
        + '\nWith the activation function :: '
        + select_activation
    console.log(model_info)

    document.getElementById('model_info').innerText = model_info;

let output;

async function train()
{

    for (let i =0 ; i < itter ; i++){

        const config={
            shuffle:true,
            epoch:epoch_user
        }
        const response = await model.fit(x, y , config) /////  fitting model to data
        
        output=  response.history.loss[0]        
        console.log('Loss :: ' , output)              
    }    
    document.getElementById('txtarea').innerText = 'The loss is :: \n' + output
    let y_pred=model.predict(x)    
    document.getElementById('oparea').innerText = 'The output is \n'+ y_pred.dataSync().join('\n')+'\n Re-run the network to get the output close to Ys or change itterations from sidebar'
    document.getElementById('txtarea').innerText = output
    
    
}
}