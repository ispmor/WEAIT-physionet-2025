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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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



logger = logging.getLogger(__name__)


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    args = ARGS()
    data_directory= data_folder
    datasets_target_dir = args.target
    gpu_number = args.gpu
    models_dir = model_folder
    clean_datasets_var=args.clean
    window_size = args.window_size
    wavelet_features_size=args.wavelet_features_size
    alpha_input_size=args.alpha_input_size
    beta_input_size=args.beta_input_size
    gamma_input_size=args.gamma_input_size
    delta_input_size=args.delta_input_size
    epsilon_input_size=args.epsilon_input_size
    zeta_input_size=args.zeta_input_size
    name = args.name
    debug_mode = verbose
    remove_baseline = args.remove_baseline
    fold_to_process = args.fold
    selected_leads_flag = args.leads
    network_name = args.network
    include_domain = args.include_domain
    alpha_hidden=args.alpha_hidden
    alpha_layers=args.alpha_layers
    beta_hidden=args.beta_hidden
    beta_layers=args.beta_layers
    epochs=args.epochs
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
    execution_time=datetime.now()
    date = execution_time.date()
    time = execution_time.time()
    log_filename =f'logs/{name}/{date}/{time}.log'
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

    headers = []
    labels = np.zeros(num_records, dtype=bool)

    record_files = []
    header_files = []

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        header_files.append(os.path.join(data_folder, get_header_file(records[i])))
        record_files.append(record)

        header = load_header(record)
        headers.append(header)
        labels[i] = get_label(header)

    totalX = list(zip(record_files, header_files))

    train_X, test_X, train_Y, test_Y = train_test_split(totalX, labels, test_size=0.25, random_state=42)

    utilityFunctions.prepare_h5_dataset(leads,  train_X, test_X, header_files, record_files)
    training_dataset = HDF5Dataset('./' + utilityFunctions.training_filename, recursive=False, load_data=False, data_cache_size=4, transform=None, leads=leads_idx)
    logger.info("Loaded training dataset")
    test_dataset = HDF5Dataset('./' + utilityFunctions.test_filename, recursive=False, load_data=False, data_cache_size=4, transform=None, leads=leads_idx)
    logger.info("Loaded validation dataset")

    model = get_MultibranchBeats(alpha_config, beta_config, gamma_config, delta_config, epsilon_config, zeta_config, utilityFunctions.all_classes,device, leads=list(leads))
    training_config = TrainingConfig(batch_size=1500,
                                    n_epochs_stop=early_stop,
                                    num_epochs=epochs,
                                    lr_rate=0.01,
                                    criterion=nn.BCEWithLogitsLoss(),
                                    optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
                                    device=device
                                    )

    training_data_loader = torch_data.DataLoader(training_dataset, batch_size=1500, shuffle=True, num_workers=6)
    validation_data_loader = torch_data.DataLoader(validation_dataset, batch_size=1500, shuffle=True, num_workers=6)
    networkTrainer=NetworkTrainer(utilityFunctions.all_classes, training_config, tensorboardWriter, "weights_eval.csv")
    trained_model_name= networkTrainer.train(model, alpha_config, beta_config, training_data_loader,  validation_data_loader, fold, leads_dict[selected_leads_flag], include_domain)




    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_model(model_folder, model)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(model_filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Load the model.
    model = model['model']

    # Extract the features.
    features = extract_features(record)
    features = features.reshape(1, -1)

    # Get the model outputs.
    binary_output = model.predict(features)[0]
    probability_output = model.predict_proba(features)[0][1]

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record):
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)

    one_hot_encoding_sex = np.zeros(3, dtype=bool)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1
    else:
        one_hot_encoding_sex[2] = 1

    signal, fields = load_signals(record)

    # TO-DO: Update to compute per-lead features. Check lead order and update and use functions for reordering leads as needed.

    num_finite_samples = np.size(np.isfinite(signal))
    if num_finite_samples > 0:
        signal_mean = np.nanmean(signal)
    else:
        signal_mean = 0.0
    if num_finite_samples > 1:
        signal_std = np.nanstd(signal)
    else:
        signal_std = 0.0

    features = np.concatenate(([age], one_hot_encoding_sex, [signal_mean, signal_std]))

    return np.asarray(features, dtype=np.float32)

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)
