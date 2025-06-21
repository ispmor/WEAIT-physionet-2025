#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
import sys
from config import ARGS
from helper_code import *
from datetime import datetime
import logging
from utilities import *
from sklearn.model_selection import train_test_split
from networks.model import BranchConfig
from torch.utils import data as torch_data
from training import *
from torch.nn import BCEWithLogitsLoss
import torch


logger = logging.getLogger(__name__)

args = ARGS()
datasets_target_dir = args.target
gpu_number = args.gpu
window_size = args.window_size
wavelet_features_size=args.wavelet_features_size
alpha_input_size=args.alpha_input_size
beta_input_size=args.beta_input_size
gamma_input_size=args.gamma_input_size
delta_input_size=args.delta_input_size
epsilon_input_size=args.epsilon_input_size
zeta_input_size=args.zeta_input_size
name = args.name
network_name = args.network
alpha_hidden=args.alpha_hidden
alpha_layers=args.alpha_layers
beta_hidden=args.beta_hidden
beta_layers=args.beta_layers
epochs=args.epochs
debug_mode=args.debug_mode
leads=('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
early_stop=args.early_stop
device = torch.device(f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu")
alpha_config = BranchConfig(network_name, alpha_hidden, alpha_layers, window_size, window_size, wavelet_features_size, beta_input_size=alpha_input_size)
beta_config = BranchConfig(network_name, alpha_hidden, alpha_layers, window_size, beta_input_size=beta_input_size)
gamma_config = BranchConfig(network_name, alpha_hidden, alpha_layers, window_size, beta_input_size=gamma_input_size, channels=1)
delta_config = BranchConfig(network_name, alpha_hidden, alpha_layers, window_size, beta_input_size=delta_input_size)
epsilon_config = BranchConfig(network_name, alpha_hidden, alpha_layers, window_size, window_size, wavelet_features_size, beta_input_size=epsilon_input_size)
zeta_config = BranchConfig(network_name, alpha_hidden, alpha_layers, window_size, window_size, wavelet_features_size, beta_input_size=zeta_input_size)

leads_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    execution_time=datetime.now()
    date = execution_time.date()
    time = execution_time.time()
    log_filename =f'logs/{name}/{date}/{time}.log'
    tensorboardWriter: SummaryWriter = SummaryWriter(f"runs/physionet-2025")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging_level = logging.INFO
    if debug_mode:
        logging_level = logging.DEBUG

    logging.basicConfig(filename=log_filename,
                      level=logging_level,
                      format='[%(asctime)s %(levelname)-8s %(filename)s:%(lineno)s]  %(message)s',
                      datefmt='%Y-%m-%d %H:%M:%S')
    logger.info(f"!!! Experiment: {name} !!!")

    utilityFunctions = UtilityFunctions(device, datasets_dir=datasets_target_dir, rr_features_size=delta_input_size, window_size=window_size, wavelet_features_size=wavelet_features_size)

    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')
        logger.debug("Finding Challenge data")

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')


    record_files = []
    header_files = []
    sami_trop_headers = []
    sami_trop_recordings = []

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])

        header = load_header(record)
        splitted = header.split("\n")
        source_info = [x for x in splitted if "Source" in x]
        if len(source_info) > 0: 
            if "SaMi-Trop" in source_info[0]:
                sami_trop_headers.append(os.path.join(data_folder, get_header_file(records[i])))
                sami_trop_recordings.append(record)     
                continue
            elif "CODE" in source_info[0]:
                if random.random() > 0.05:
                    continue
            else:
                if random.random() > 0.30:
                    continue

        
        header_files.append(os.path.join(data_folder, get_header_file(records[i])))
        record_files.append(record)
        
    totalX = list(zip(record_files, header_files))
    totalSami = list(zip(sami_trop_recordings, sami_trop_headers))

    labels = np.zeros(len(totalX), dtype=bool)

    train_X, test_X, _, _ = train_test_split(totalX, labels, test_size=0.3, random_state=42)
    sami_trop_training, sami_trop_test, _, _ = train_test_split(totalSami, np.ones(len(sami_trop_recordings)), test_size=0.3, random_state=42)

    train_X.extend(sami_trop_training)
    test_X.extend(sami_trop_test)

    del labels

    utilityFunctions.prepare_h5_dataset(leads,  train_X, test_X)
    del train_X, test_X
    training_dataset = HDF5Dataset('./' + utilityFunctions.training_filename, recursive=False, load_data=False, data_cache_size=4, transform=None, leads=leads_idx)
    logger.info("Loaded training dataset")
    weights = utilityFunctions.load_training_weights()

    test_dataset = HDF5Dataset('./' + utilityFunctions.test_filename, recursive=False, load_data=False, data_cache_size=4, transform=None, leads=leads_idx)
    logger.info("Loaded validation dataset")

    model = get_MultibranchBeats(alpha_config, beta_config, gamma_config, delta_config, epsilon_config, utilityFunctions.all_classes,device, leads=list(leads))
    training_config = TrainingConfig(batch_size=500,
                                    n_epochs_stop=early_stop,
                                    num_epochs=epochs,
                                    lr_rate=0.01,
                                    criterion=BCEWithLogitsLoss(pos_weight=weights),
                                    optimizer=torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01),
                                    device=device,
                                    model_repository=model_folder
                                    )

    training_data_loader = torch_data.DataLoader(training_dataset, batch_size=500, shuffle=True, num_workers=6)
    test_data_loader = torch_data.DataLoader(test_dataset, batch_size=500, shuffle=True, num_workers=6)
    networkTrainer=NetworkTrainer(utilityFunctions.all_classes, training_config, tensorboardWriter)
    trained_model_name= networkTrainer.train(model, alpha_config, beta_config, training_data_loader,  test_data_loader, leads)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    checkpoint = torch.load(os.path.join(model_folder, "best_model_physionet2025.th"), map_location=torch.device(device))
    model = get_MultibranchBeats(alpha_config, beta_config, gamma_config,
                                 delta_config, epsilon_config, 
                                 ['True'],device, leads=leads)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.leads = checkpoint['leads']
    model = model.to(device)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    execution_time=datetime.now()
    date = execution_time.date()
    time = execution_time.time()
    log_filename =f'logs/{name}/{date}/{time}-TEST.log'
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging_level = logging.INFO
    if debug_mode:
        logging_level = logging.DEBUG

    logging.basicConfig(filename=log_filename,
                      level=logging_level,
                      format='[%(asctime)s %(levelname)-8s %(filename)s:%(lineno)s]  %(message)s',
                      datefmt='%Y-%m-%d %H:%M:%S')
    logger.info(f"!!! Experiment: {name} !!!")

    utilityFunctions = UtilityFunctions(device, rr_features_size=delta_input_size, window_size=window_size, wavelet_features_size=wavelet_features_size)
    # Extract the features.
    header = load_header(record)
    header_file = record + ".hea"
    print(f"Processing {header_file}")

    try:
        age, sex_one_hot_encoding, source, signal_mean, signal_std, signal, fields = utilityFunctions.extract_features(record)
    except Exception as e:
        print(f"Skipping {header_file} and associated recording  because of {e}")
        return 0.0, 0.0

    recording_full=utilityFunctions.load_and_equalize_recording(signal, fields, header_file, 400)
    drift_removed_recording, recording, signals, infos, peaks, rates = utilityFunctions.preprocess_recording(recording_full, header,  leads_idxs=utility_functions.leads_idxs_dict[len(leads)])
            
    recording_features_record = np.concatenate((age, sex_one_hot_encoding, signal_mean, signal_std))

    if signals is None or infos is None or peaks is None or rates is None:
        print("Failed to extract needed data - returning zeros")
        probability_output = 0.0
        binary_output = 0.0
        return binary_output, probability_output


    recording_raw, recording_drift_removed, rr_features, wavelet_features= utilityFunctions.one_file_training_data(recording, drift_removed_recording, signals, infos, rates, utilityFunctions.window_size, peaks, header, leads)

    recording_features = torch.Tensor(np.array([recording_features_record] * recording_raw.shape[0])).to(device)

    recording_raw = torch.Tensor(recording_raw).to(device)
    recording_drift_removed = torch.Tensor(recording_drift_removed).to(device)
    rr_features = torch.Tensor(rr_features).to(device)
    wavelet_features = torch.Tensor(wavelet_features).to(device)

    batch = (recording_raw, recording_drift_removed,  None, rr_features, wavelet_features, recording_features)
    alpha_input, beta_input, gamma_input, delta_input, epsilon_input, recording_features, _= batch_preprocessing(batch)
    # Get the model outputs.
    try:
        model_outputs = model(alpha_input, beta_input, gamma_input, delta_input, epsilon_input, recording_features)
        probability_output = torch.mean(torch.nn.functional.sigmoid(model_outputs), 0).detach().cpu().numpy()[0]
        binary_output = float(probability_output > 0.5)
        return binary_output, probability_output
    except Exception as e:
        print(f"Model failed  because of {e}")
        logger.debug(f"Failed to run model because of {e}",exc_info=True)
        return 0.0, 0.0

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)
