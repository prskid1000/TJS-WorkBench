import React from 'react';
import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis';
import {Container, FormControl, Button} from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
/*eslint-disable */

var logger = "";

function plot(d)
{
  const series = ['trend'];
  var dat = [];
  for(var i = 0; i < d.length; i++)
  {
    var obj ={"x":d[i][0], "y":d[i][1]};
    dat.push(obj);
  }
  console.log(dat);
  const data = { values: [dat],  series};
  const surface = { name: 'Line chart', tab: 'Charts' };
  tfvis.render.linechart(surface, data);
  tfvis.visor().open();
}

function createModel(layer, neuron) {
  const model = tf.sequential();
  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
  for(var i = 0; i< parseInt(layer); i++)
  model.add(tf.layers.dense({units: parseInt(neuron), useBias: true, activation:"sigmoid"}));
  model.add(tf.layers.dense({units: 1, useBias: true}));
  model.compile({optimizer: tf.train.adam(),loss: tf.losses.meanSquaredError,metrics: ['mse'],
  });
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
      inputs.push([parseFloat(data[i][0])]);
      labels.push([parseFloat(data[i][1])]);
    }

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

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
        aria-describedby="basic-addon1"
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
            [0,0],
            [0,0],
            [0,0],
        ],
        sample:[
          [0,0]
        ],
        prediction:"",
        epoch:0,
        layer:0,
        neuron:0
      };
      this.epochref = React.createRef();
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
          console.log(this.state.sample);
        }

    render() {

      return (
        <Container className="container">
        <h1><center>Regressor</center></h1>
        <br/><br/><br/>
        <table>
        <thead>
        <tr>
        <th>Independent Variable</th>
        <th>Dependent Variable</th>
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
        <th>
        <Button variant="success" onClick={()=>{
          this.state.data.push([0,0]);
          this.setState({data:this.state.data});
        }}>Add Row</Button>
        &nbsp;
        <Button variant="success" onClick={()=>{
          plot(this.state.data);
        }}>Visualize</Button>
        </th>
        </tr>
        </tbody>
        </table>

        <br/>
        <table>
        <tbody>
        <tr>
        <th><label>No. of epochs</label></th>
        <th><label>No. of Layers</label></th>
        <th><label>Neuron in each layer</label></th>
        </tr>
        <tr>
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
        <th>
        <Button variant="success" onClick={()=>{
          model = createModel(this.state.layer,this.state.neuron);
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
        }}>Train</Button>
        </th>
        </tr>
        </tbody>
        </table>
        <br/>
        <label><b>Test Model</b></label>
        <table>
        <tbody>
        <tr>
        <th><Cell i = "0" j = "0" onChange={this.eventCell2}/></th>
        <th><Button variant="success" onClick={async ()=>{
          const {labelMin, labelMax} = convertToTensor(this.state.data);
          const {inputs} = await  convertToTensor(this.state.sample);
          var pred = await model.predict(inputs).mul(labelMax.sub(labelMin)).add(labelMin).array();
          this.setState({prediction:parseFloat(pred[0])});
        }}>Predict</Button></th>
        <th><FormControl
          placeholder={this.state.prediction}
          aria-label=""
          aria-describedby="basic-addon1"
          onChange={()=>{}}/></th>
        </tr>
        </tbody>
        </table>
        </Container>
      );
    }
}

export default App;
