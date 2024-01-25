# stdlib
import argparse
import configparser
import os
from datetime import datetime

# third party
import numpy as np
import tensorflow as tf
import torch
from models import MLP
from optimizers import init_optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Change config filename based on input."
    )

    # Specify a default value for the 'input' argument
    parser.add_argument(
        "-c",
        "--config",
        help="Input to change the config name.",
        default="../configs/pygranso.cfg",
    )

    args = parser.parse_args()
    print("Loading cfgs")

    cfg = configparser.ConfigParser()
    # cp.read('./config/config_mnist.cfg')
    cfg.read(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BASE_FOLDER = cfg.get('EXP', 'FOLDER')

    OPT_NAME = cfg.get('OPTIMIZER', 'NAME')
    DATASET_NAME = cfg.get('DATASET', 'NAME')
    DATASET_PATH = cfg.get('DATASET', 'PATH')
    npz_data = np.load(DATASET_PATH)

    now = datetime.now()

    # Format the date and time as a string
    date_time_str = now.strftime("%Y%m%d_%H%M%S")

    NUM_FEATURES = npz_data['train_X_0'].shape[1]
    THRESHOLD = cfg.get('EXP', 'THRESHOLD')
    EXP_FOLDER = os.path.join(
        BASE_FOLDER,
        'exp',
        f'r_{THRESHOLD}_{date_time_str}',
        f'{OPT_NAME}_{DATASET_NAME}',
    )
    os.makedirs(EXP_FOLDER, exist_ok=True)

    if OPT_NAME == 'PyGRANSO':
        END_REPEAT = cfg.getint('EXP', 'END_REPEAT')
        START_REPEAT = cfg.getint('EXP', 'START_REPEAT')

        MODEL_TYPE = cfg.get('MODEL', 'TYPE')
        LAYER_WIDTH = cfg.getint('MODEL', 'LAYER_WIDTH')
        NUM_LAYERS = cfg.getint('MODEL', 'NUM_LAYERS')
        MODEL_PATH = cfg.get('MODEL', 'PATH')
        MODEL_OUT_ACTIVATION = cfg.get('MODEL', 'OUT_ACTIVATION')

        data = {key: torch.tensor(npz_data[key]) for key in npz_data.files}

        print('\n' + '=' * 30)
        print('Building Model')
        print('Optimizer is PyGRANSO, building pytorch model')
        MODEL_DIR = os.path.join(MODEL_PATH, f'{MODEL_TYPE}_Feature{NUM_FEATURES}')

        model = MLP(1, NUM_FEATURES, NUM_LAYERS, LAYER_WIDTH, MODEL_OUT_ACTIVATION)
        for i in range(START_REPEAT, END_REPEAT):
            print('\n' + '=' * 30)
            print(f'EXP {i} Start!!!')
            weight_path = os.path.join(MODEL_DIR, f'model_pytorch_{i}.pt')
            fn = os.path.join(EXP_FOLDER, f'{i:06}.npz')
            print(f'>Results will be save to {fn}')
            print(f'>Loading weight from: {weight_path}')
            model.load_state_dict(torch.load(weight_path))
            opt = init_optimizer(cfg, data, device, model, fn=fn)
            opt.train()

    elif OPT_NAME == 'TFCO':
        END_REPEAT = cfg.getint('EXP', 'END_REPEAT')
        START_REPEAT = cfg.getint('EXP', 'START_REPEAT')

        MODEL_TYPE = cfg.get('MODEL', 'TYPE')
        LAYER_WIDTH = cfg.getint('MODEL', 'LAYER_WIDTH')
        NUM_LAYERS = cfg.getint('MODEL', 'NUM_LAYERS')
        MODEL_PATH = cfg.get('MODEL', 'PATH')
        MODEL_OUT_ACTIVATION = cfg.get('MODEL', 'OUT_ACTIVATION')

        data = {key: torch.tensor(npz_data[key]) for key in npz_data.files}

        print('\n' + '=' * 30)
        print('Building Model')
        print('Optimizer is TFCO, building tensorflow model')
        MODEL_DIR = os.path.join(MODEL_PATH, f'{MODEL_TYPE}_Feature{NUM_FEATURES}')

        # model = create_tf_model(1,NUM_FEATURES,NUM_LAYERS,LAYER_WIDTH,MODEL_OUT_ACTIVATION)  # noqa

        for i in range(START_REPEAT, END_REPEAT):
            print('\n' + '=' * 30)
            print(f'EXP {i} Start!!!')
            weight_path = os.path.join(MODEL_DIR, f'model_keras_{i}.h5')
            fn = os.path.join(EXP_FOLDER, f'{i:06}.npz')
            print(f'>Results will be save to {fn}')
            print(f'>Loading weight from: {weight_path}')
            # model = tf.saved_model.load(weight_path)
            model = tf.keras.models.load_model(weight_path)
            # model.summary()
            # model.load_weights(weight_path)
            opt = init_optimizer(cfg, data, device, model, fn=fn)
            opt.train()
