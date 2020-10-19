import React from 'react';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import {Container, FormControl, Button} from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
/*eslint-disable */

var logger = "";

function heatmap(d)
{
  var dat = [];
  for(var i = 0; i < d.length; i++) dat.push([d[i][0],d[i][1],d[i][2]]);
  const data = {
   values: dat,
   xTickLabels: ['Feature-1', 'Feature-2', 'Feature-3'],
   yTickLabels: ['Feature-1', 'Feature-2', 'Feature-3'],
 }

 const surface = { name: 'Heatmap', tab: 'Charts'};
 tfvis.render.heatmap(surface, data);
 tfvis.visor().open();
}

function createModel(learn, layer, neuron) {
  const model = tf.sequential();
  model.add(tf.layers.dense({inputShape: [3], units: 3, useBias: true}));
  for(var i = 0; i< parseInt(layer); i++)
  model.add(tf.layers.dense({units: parseInt(neuron), useBias: true, activation:"sigmoid"}));
  model.add(tf.layers.dense({activation:"softmax", units:2}));
  model.compile({loss:"categoricalCrossentropy",metrics:['accuracy','mse'], optimizer:tf.train.adam(learn)});
  tfvis.visor().open();
  return model;
}

async function trainModel(ref, model, inputs, labels, epoch) {

  const batchSize = 5;
  const epochs = epoch;
  tfvis.visor().open();

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
   { name: 'Training Performance' },
   ['loss', 'mse', 'accuracy'],
   { height: 200, callbacks: ['onEpochEnd'] })
  });
}

function convertToTensor(data) {

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);

    const inputs = []
    const labels = []

    for(var i = 0; i < data.length; i++)
    {
      inputs.push([parseFloat(data[i][0]), parseFloat(data[i][1]), parseFloat(data[i][2])]);
      labels.push([parseFloat(data[i][3]), parseFloat(data[i][4])]);
    }

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 3]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 2]);

    //console.log(labels);

    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });
}

var model;
class Cell extends React.Component
{
  constructor(props){
      super(props);
      this.state =
      {
        data:"",
        i:this.props.i,
        j:this.props.j
      }
      this.handleChange = this.handleChange.bind(this);
  }

  handleChange(event) {
    this.setState({data:event.target.value},() => {
      if (this.props.onChange) {
        this.props.onChange(this.state);
      }
    });
    }

  render() {
    return (
      <FormControl
        placeholder=""
        aria-label=""
        onChange={this.handleChange}
      />
    );
  }
}

class App extends React.Component
{
    constructor(props){
        super(props);
        this.state =
        {
          value:"",
          data:[
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
        ],
        sample:[
          [0,0,0,0,0]
        ],
        prediction:"",
        learn:0,
        epoch:0,
        layer:0,
        neuron:0
      };
        this.epochref = React.createRef();
        this.learnref = React.createRef();
        this.layerref = React.createRef();
        this.neuronref = React.createRef();
    }

    eventCell = data => {
        this.state.data[data.i][data.j] = data.data;
        this.setState({data:this.state.data});
        //console.log(this.state.data);
      }

      eventCell2 = data => {
          this.state.sample[data.i][data.j] = data.data;
          this.setState({sample:this.state.sample});
          //console.log(this.state.data);
        }

    render() {

      return (
        <Container className="container">
        <h1><center>Classifier</center></h1>
        <br/><br/><br/>
        <table>
        <thead>
        <tr>
        <th>Feature-1</th>
        <th>Feature-2</th>
        <th>Feature-3</th>
        <th>Class-1</th>
        <th>Class-2</th>
        </tr>
        </thead>
        <tbody>
        {this.state.data.map((itemi,i) => (
          <tr key ={i}>
            {itemi.map((itemj,j) => (<th key ={i + j}><Cell i = {i} j = {j} onChange={this.eventCell}/></th>))}
          </tr>
        ))}
        <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th><Button variant="success" onClick={()=>{
          this.state.data.push([0,0,0,0,0]);
          this.setState({data:this.state.data});
        }}>Add Row</Button>
        &nbsp;
        <Button variant="success" onClick={()=>{
          heatmap(this.state.data);
        }}>Visualize</Button></th>
        </tr>
        </tbody>
        </table>

        <table>
        <tbody>
        <tr>
        <th><label>Learning Rate</label></th>
        <th><label>No. of epochs</label></th>
        <th><label>No. of Layers</label></th>
        <th><label>Neuron in each layer</label></th>
        </tr>
        <tr>
        <th>
        <FormControl
          ref={this.learnref}
          placeholder={this.state.learn}
          aria-label=""
          onChange={()=>{this.setState({learn:this.learnref.current.value})}}/>
        </th>
        <th>
        <FormControl
          ref={this.epochref}
          placeholder={this.state.epoch}
          aria-label=""
          onChange={()=>{this.setState({epoch:this.epochref.current.value})}}/>
        </th>
        <th>
        <FormControl
          ref={this.layerref}
          placeholder={this.state.layer}
          aria-label=""
          onChange={()=>{this.setState({layer:this.layerref.current.value})}}/>
        </th>
        <th>
        <FormControl
          ref={this.neuronref}
          placeholder={this.state.neuron}
          aria-label=""
          onChange={()=>{this.setState({neuron:this.neuronref.current.value})}}/>
        </th>
        </tr>
        <tr>
        <th></th>
        <th></th>
        <th></th>
        <th>
        <Button variant="success" onClick={()=>{
          model = createModel(this.state.learn,this.state.layer,this.state.neuron);
          const surface1 = { name: 'Model Summary', tab: 'Model Inspection'};
          const surface2 = { name: 'Layer Summary', tab: 'Model Inspection'};
          tfvis.show.layer(surface2, model.getLayer(undefined, 1));
          tfvis.show.modelSummary(surface1, model);
          tfvis.visor().open();
        }}>Model</Button>
        &nbsp;
        <Button variant="success" onClick={()=>{
          const tensorData = convertToTensor(this.state.data);
          const {inputs, labels} = tensorData;
          trainModel(this, model, inputs, labels, this.state.epoch);
          tfvis.visor().open();
        }}>Train</Button>
        </th>
        </tr>
        </tbody>
        </table>

        <br/><br/>
        <label><b>Test Model</b></label>
        <table>
        <tbody>
        <tr>
        <th><Cell i = "0" j = "0" onChange={this.eventCell2}/></th>
        <th><Cell i = "0" j = "1" onChange={this.eventCell2}/></th>
        <th><Cell i = "0" j = "2" onChange={this.eventCell2}/></th>
        <th><Button variant="success" onClick={async ()=>{
          const tensorData = convertToTensor(this.state.sample);
          const {inputs,} = tensorData;
          var pred = await model.predict(inputs).array();
          //console.log(pred);
          if(parseFloat(pred[0][0]) > parseFloat(pred[0][1]))
          {
            this.setState({prediction:"Class-1"});
          }
          else {
            this.setState({prediction:"Class-2"});
          }
        }}>Predict</Button></th>
        <th><FormControl
          placeholder={this.state.prediction}
          aria-label=""
          onChange={()=>{}}/></th>
        </tr>
        </tbody>
        </table>
        </Container>
      );
    }
}

export default App;
