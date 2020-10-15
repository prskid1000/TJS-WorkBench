import React from 'react';
import * as tf from '@tensorflow/tfjs'
import {Container, FormControl, Button} from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

var logger = "";

function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({inputShape: [3], units: 3, useBias: true}));
  model.add(tf.layers.dense({units: 50, useBias: true}));
  model.add(tf.layers.dense({activation:"softmax", units:2}));
  model.compile({loss:"categoricalCrossentropy",metrics:['accuracy','mse'], optimizer:tf.train.adam(0.06)});
  return model;
}

async function trainModel(ref, model, inputs, labels) {

  const batchSize = 5;
  const epochs = 100;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: {
      onEpochEnd: async(epoch,logs) =>{
        logger = "EPOCH: " + epoch + "    LOSS: " + logs.loss + "    MSE: " + logs.mse + "    ACCURACY: " + logs.acc;
        ref.state.value = logger;
        ref.setState({value:logger});
      }
    }
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

var model = createModel();

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
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
        ],
        sample:[
          [0,0,0,0,0]
        ],
        prediction:""
        }
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
        }}>Add Row</Button></th>
        </tr>
        </tbody>
        </table>
        <table>
        <tbody>
        <tr>
        <th><textarea className="form-control" value={this.state.value} rows="1" cols="100" onChange={()=>{}}/></th>
        <th><Button variant="success" onClick={()=>{
          const tensorData = convertToTensor(this.state.data);
          const {inputs, labels} = tensorData;
          trainModel(this, model, inputs, labels);
        }}>Train</Button></th>
        </tr>
        </tbody>
        </table>
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
