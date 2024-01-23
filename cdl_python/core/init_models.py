import torch
import torch.nn as nn
import h5py
import os

from models import MLP, create_tf_model
import numpy as np

from pt2keras import Pt2Keras
from pt2keras import converter

import argparse
import configparser

import tensorflow as tf
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Change config filename based on input.")
    
    # Specify a default value for the 'input' argument.
    parser.add_argument("-c", "--config", 
                        help="Input to change the config name.", 
                        default="../configs/pygranso.cfg")
    
    args = parser.parse_args()
    print("Loading cfgs")

    cfg = configparser.ConfigParser()
    #cp.read('./config/config_mnist.cfg')
    cfg.read(args.config)

    DATASET_PATH = cfg.get('DATASET','PATH')
    data = np.load(DATASET_PATH)

    NUM_FEATURES = data['train_X_0'].shape[1]
    print(NUM_FEATURES)

    END_REPEAT = cfg.getint('EXP','END_REPEAT')
    START_REPEAT = cfg.getint('EXP','START_REPEAT')

    MODEL_TYPE = cfg.get('MODEL','TYPE')
    LAYER_WIDTH = cfg.getint('MODEL','LAYER_WIDTH')
    NUM_LAYERS = cfg.getint('MODEL','NUM_LAYERS')
    MODEL_PATH = cfg.get('MODEL','PATH')
    MODEL_OUT_ACTIVATION = cfg.get('MODEL','OUT_ACTIVATION')

    MODEL_DIR = os.path.join(MODEL_PATH,f'{MODEL_TYPE}_Feature{NUM_FEATURES}')
    os.makedirs(MODEL_DIR,exist_ok=True)

    # Initialize and save the model weights
    for i in range(START_REPEAT,END_REPEAT):  # Example for 3 initializations
        print(f'Generating init weight {i}...')
        if(MODEL_TYPE == 'MLP'):
            model = MLP(1,NUM_FEATURES,NUM_LAYERS,LAYER_WIDTH,MODEL_OUT_ACTIVATION)
            torch.save(
                model.state_dict(), 
                os.path.join(MODEL_DIR,f'model_pytorch_{i}.pt')
                )
            tf_fn = os.path.join(MODEL_DIR,f'model_keras_{i}.h5')
            # Convert and save as .h5 file for TensorFlow

            #onnx_fn = os.path.join(MODEL_DIR,f'onnx_model_{i}.onnx')
            #dummy_input = torch.randn(1, NUM_FEATURES)
            model.eval()  # Set the model to inference mode

            # Export the model
            #torch.onnx.export(model,
            #      dummy_input,
            #      onnx_fn,
            #      export_params=True,
            #      opset_version=11,  # or the version best suited for your needs
            #      do_constant_folding=True,  # whether to execute constant folding for optimization
            #      input_names=['input'],  # name your input
            #      output_names=['output'] # name your output
            #      #dynamic_axes={'input': {0: 'batch_size'},  # if your model supports dynamic batch size
            #      #              'output': {0: 'batch_size'}})
            #)

            # Load the ONNX file
            #model_onnx = onnx.load(onnx_fn)
            #for i, node in enumerate(model_onnx.graph.node):
            #    node.name = node.name.replace('::', '__')  # Replace problematic characters

            # Import the ONNX model to Tensorflow
            #for node in model_onnx.graph.node:
            #    # Replace '::' with '__' or some other placeholder
            #    node.name = node.name.replace('::', '__')
            #keras_model = onnx_to_keras(model_onnx,['input'])
            keras_model = create_tf_model(1,NUM_FEATURES,NUM_LAYERS,LAYER_WIDTH,MODEL_OUT_ACTIVATION)
            #keras_model.summary()
            #for layer in keras_model.layers:
            #    print(layer.name)
            k = 0
            
            for i, layer in enumerate(model.layers):
                #print(layer)
                
                if isinstance(layer, nn.Linear):
                    #print(i-k)
                    #print(keras_model.layers[i-k].name)
                    weight, bias = layer.weight.data.numpy(), layer.bias.data.numpy()
                    keras_model.layers[i-k].set_weights([weight.T, bias])  # Multiply by 3 because of batch norm and activation layers
                elif isinstance(layer, nn.BatchNorm1d):
                    #print(i-k)
                    #print(keras_model.layers[i-k].name)
                    gamma, beta = layer.weight.data.numpy(), layer.bias.data.numpy()
                    mean, var = layer.running_mean.data.numpy(), layer.running_var.data.numpy()
                    keras_model.layers[i-k].set_weights([gamma, beta, mean, var])
                else:
                    k +=1
            
            input_data = np.random.rand(10,NUM_FEATURES).astype(np.float32)
            pytorch_output = model(torch.from_numpy(input_data).float()).detach().numpy()
            keras_output = keras_model.predict(input_data)
            print(pytorch_output)
            print(keras_output)
            print(np.sum(np.abs(pytorch_output - keras_output)))
            #print(np.allclose(pytorch_output, keras_output, atol=1e-5))
            keras_model.save(tf_fn)
    print('Done.')