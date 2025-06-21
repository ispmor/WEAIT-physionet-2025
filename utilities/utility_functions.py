import math
import h5py
import neurokit2 as nk
from networks.model import  get_MultibranchBeats
from pywt import wavedec
from helper_code import *
from .pan_tompkins_detector import *
from .results_handling import *
from .data_preprocessing import *
import numpy as np
import logging
import torch
import csv
from .domain_knowledge_processing import analyse_recording, analysis_dict_to_array
from .raw_signal_preprocessing import baseline_wandering_removal, wavelet_threshold, remove_baseline_drift
from scipy.signal import resample
import gc
import random

logger = logging.getLogger(__name__)

thrash_data_dir="../data/irrelevant"

leads_idxs = {'I': 0, 'II': 1, 'III':2, 'aVR': 3, 'aVL':4, 'aVF':5, 'V1':6, 'V2':7, 'V3':8, 'V4':9, 'V5':10, 'V6':11}
leads_idxs_dict = {
        12: {'I': 0, 'II': 1, 'III':2, 'aVR': 3, 'aVL':4, 'aVF':5, 'V1':6, 'V2':7, 'V3':8, 'V4':9, 'V5':10, 'V6':11},
        } #we should use aVF instead of II in 2 leads example



def save_headers_recordings_to_json(filename, headers, recordings, idxs):
    with open(filename, 'w') as f:
        data = {
                "header_files": list(np.array(headers)[idxs]),
                "recording_files":list(np.array(recordings)[idxs]),
                }
        json.dump(data, f)





class UtilityFunctions:
    all_classes = ['True']
    classes_counts = dict(zip(all_classes, [0,0]))
    device=None
    twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
    four_leads = ('I', 'II', 'III', 'V2')
    three_leads = ('I', 'II', 'V2')
    two_leads = ('I', 'II')
    leads_set = [twelve_leads]#, six_leads, four_leads, three_leads, two_leads]

    classes = set()



    def __init__(self, device, datasets_dir="h5_datasets/", window_size=1500, rr_features_size=16, wavelet_features_size=185) -> None:
        self.device = device
        self.window_size = window_size
        self.rr_features_size = rr_features_size
        self.wavelet_features_size = wavelet_features_size

        self.training_filename = datasets_dir + 'cinc_database_training.h5'
        self.validation_filename = datasets_dir + 'cinc_database_validation_{0}_{1}.h5'
        self.training_full_filename = datasets_dir + 'cinc_database_training_full_{0}_{1}.h5'
        self.test_filename = datasets_dir + 'cinc_database_test.h5'
        self.training_weights_filename = datasets_dir + "weights.csv" 

        logger.debug(f"Initiated UtilityFunctions: {self.__dict__}")




    def prepare_h5_dataset(self, leads, single_fold_data_training, single_fold_data_test):
        print(f"Preparing HDF5 dataset from WFDB files")
        training_data_length = len(single_fold_data_training)
        test_data_length = len(single_fold_data_test)

        training_filename = self.training_filename
        test_filename = self.test_filename


        if not os.path.isfile(training_filename):
            print(f"{training_filename} not found, creating database")
            positive_class_count, negative_class_count = self.create_hdf5_db(training_data_length, single_fold_data_training,  leads, isTraining=1, filename=training_filename)
            np.savetxt(self.training_weights_filename, np.asarray(np.vstack([positive_class_count, negative_class_count])), delimiter=',')



        if not os.path.isfile(test_filename):
            print(f"{test_filename} not found, creating database")
            _, _ = self.create_hdf5_db(test_data_length,  single_fold_data_test, leads, isTraining=0, filename=test_filename)





    def equalize_signal_frequency(self, freq, recording_full):
        new_recording_full = []
        if freq != 400:
            current_length = len(recording_full[0])
            scaler = float(400 / freq)
            target_samples = int(math.ceil(current_length * scaler))
            for lead in recording_full:
                new_lead = resample(lead, target_samples)
                new_recording_full.append(new_lead)
            new_recording_full = np.array(new_recording_full)

        return new_recording_full




    def one_file_training_data(self, recording, drift_removed_recording, signals, infos, rates, single_peak_length, peaks, header_file, leads):

        x_raw = []
        x_drift_removed = []
        x_baseline_removed = []
        coeffs = []
        peaks_considered = []
        recording_length=len(drift_removed_recording[0])
        for peak in range(0, recording_length, self.window_size):
            if peak + self.window_size < recording_length:
                signal_local = drift_removed_recording[:, peak: peak + self.window_size]
                signal_local_raw = recording[:, peak: peak + self.window_size]
                wavelet_features = self.get_wavelet_features(signal_local, 'db2')
                peaks_considered.extend([p for p in peaks if peak <= p < peak+self.window_size])
            else:
                logger.debug(f"Skipping append as peak = {peak}")
                continue

            logger.debug(f"Adding to X_features: {signal_local}")
            x_drift_removed.append(signal_local)
            x_raw.append(signal_local_raw)
            coeffs.append(wavelet_features)

        x_raw = np.array(x_raw, dtype=np.float64)
        x_drift_removed = np.array(x_drift_removed, dtype=np.float64)
        x_baseline_removed = np.array(x_baseline_removed, dtype=np.float64)
        coeffs = np.nan_to_num(np.asarray(coeffs,  dtype=np.float64))

        rr_features = np.zeros((x_drift_removed.shape[0], drift_removed_recording.shape[0], self.rr_features_size), dtype=np.float64)
        counter = 0 
        for peak in range(0, recording_length-self.window_size, self.window_size):

            try:
                domain_knowledge_analysis = analyse_recording(drift_removed_recording, signals, infos, rates,leads_idxs_dict[len(leads)], window=(peak, peak+self.window_size), pantompkins_peaks=peaks_considered)
                logger.debug(f"{domain_knowledge_analysis}")
                rr_features[counter] = analysis_dict_to_array(domain_knowledge_analysis, leads_idxs_dict[len(leads)])
                counter += 1

            except Exception as e:
                logger.warning(f"Currently processed file: {header_file}, issue:{e}", exc_info=True)
                raise


        return x_raw, x_drift_removed, rr_features, coeffs



    def get_wavelet_features(self, signal, wavelet):
        #TODO WHy do I downsample the signal ?!
        a4, d4, d3, d2, d1 = wavedec(signal[:, ::2], wavelet, level=4)
        return np.hstack((a4, d4, d3, d2, d1))



    def preprocess_recording(self, recording, header, leads_idxs, denoise_wavelet="db6", deniose_level=3, peaks_method="pantompkins1985", sampling_rate=400):
        if recording is None:
            return (None, None, None, None, None, None)
        drift_removed_recording, _ = remove_baseline_drift(recording)
        signals = {}
        infos = {}
        rpeaks_avg = []
        rates = {}
        was_logged=False
        for lead_name, idx in leads_idxs.items():
            coeffs = wavedec(data=drift_removed_recording[idx], wavelet=denoise_wavelet, level=deniose_level)
            drift_removed_recording[idx] = wavelet_threshold(drift_removed_recording[idx], coeffs, denoise_wavelet)
            rpeaks, signal, info = None, None, None
            try:
                rpeaks = nk.ecg_findpeaks(drift_removed_recording[idx], sampling_rate, method=peaks_method)
                signal, info =nk.ecg_delineate(drift_removed_recording[idx], rpeaks=rpeaks, sampling_rate=sampling_rate, method='dwt')
            except Exception as e:
                if not was_logged:
                    logger.warning(e, exc_info=True)
                    logger.warning(f"Comming from: \n{header}")
                    was_logged=True

            signals[lead_name] = signal
            infos[lead_name] = info
            if rpeaks is not None and info is not None:
                rpeaks_avg.append(rpeaks['ECG_R_Peaks'])
                info['ECG_R_Peaks'] = rpeaks['ECG_R_Peaks']
                rates[lead_name] = nk.ecg_rate(rpeaks, sampling_rate=sampling_rate)

        recording = np.nan_to_num(recording)
        drift_removed_recording = np.nan_to_num(drift_removed_recording)
        if len(rpeaks_avg) > 0:
            min_length = min([len(x) for x in rpeaks_avg])
            rpeaks_avg = np.array([rpeaks_avg[i][ :min_length] for i in range(len(rpeaks_avg))])
            peaks = np.mean(rpeaks_avg[:, ~np.any(np.isnan(rpeaks_avg), axis=0)], axis=0).astype(int)
            logger.debug(f"Peaks: {peaks}")

            return (drift_removed_recording, recording, signals, infos, peaks, rates)
        else:
            return (drift_removed_recording, recording, signals, infos, None, None)


    def load_and_equalize_recording(self, signal, fields, header_file, sampling_rate):
        try:
            if len(signal) > self.window_size * 10:
                signal = signal[:self.window_size * 10]

            signal = np.transpose(signal)
            recording = np.array(signal, dtype=np.float32)
            freq = fields["fs"]
            if freq != float(sampling_rate):
                recording = self.equalize_signal_frequency(freq, recording) 
        except Exception as e:
            logger.warning(f"Skipping {header_file} and associated recording  because of {e}", exc_info=True)
            recording = None

        return recording

    def extract_features(self, record: str):
        header = load_header(record)

        # Extract the age from the record.
        age = get_age(header)
        age = np.array([age])

        # Extract the sex from the record and represent it as a one-hot encoded vector.
        sex = get_sex(header)
        sex_one_hot_encoding = np.zeros(3, dtype=bool)
        if sex.casefold().startswith('f'):
            sex_one_hot_encoding[0] = 1
        elif sex.casefold().startswith('m'):
            sex_one_hot_encoding[1] = 1
        else:
            sex_one_hot_encoding[2] = 1

        # Extract the source from the record (but do not use it as a feature).
        source = get_source(header)

        # Load the signal data and fields. Try fields.keys() to see the fields, e.g., fields['fs'] is the sampling frequency.
        signal, fields = load_signals(record)
        channels = fields['sig_name']

        # Reorder the channels in case they are in a different order in the signal data.
        reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        num_channels = len(reference_channels)
        signal = reorder_signal(signal, channels, reference_channels)

        # Compute two per-channel features as examples.
        signal_mean = np.zeros(num_channels)
        signal_std = np.zeros(num_channels)

        for i in range(num_channels):
            num_finite_samples = np.sum(np.isfinite(signal[:, i]))
            if num_finite_samples > 0:
                signal_mean[i] = np.nanmean(signal)
            else:
                signal_mean = 0.0
            if num_finite_samples > 1:
                signal_std[i] = np.nanstd(signal)
            else:
                signal_std = 0.0 
                #TODO this is a potentail bug when it comes to dimensions

        # Return the features.

        return age, sex_one_hot_encoding, source, signal_mean, signal_std, signal, fields





    def create_hdf5_db(self, num_recordings, data, leads, isTraining = 1, filename=None, sampling_rate=400):
        group = None
        if isTraining == 1:
            group = 'training'
        else :
            group = 'test'

        if not filename:
            filename = f'cinc_database_{group}.h5'

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pos_signals = 1
        neg_signals = 1

        h5file = h5py.File(filename, 'w')
        grp = h5file.create_group(group)
        dset = grp.create_dataset("data", (1, len(leads), self.window_size), maxshape=(None, len(leads), self.window_size), dtype='f', chunks=(1, len(leads), self.window_size))
        lset = grp.create_dataset("label", (1,1), maxshape=(None, 1), dtype='f', chunks=(1, 1))
        recording_features = grp.create_dataset("recording_features", (1,28), maxshape=(None, 28), dtype='f', chunks=(1, 28))
        rrset = grp.create_dataset("rr_features", (1, len(leads), self.rr_features_size), maxshape=(None, len(leads), self.rr_features_size), dtype='f', chunks=True)
        waveset = grp.create_dataset("wavelet_features", (1, len(leads), self.wavelet_features_size), maxshape=(None, len(leads), self.wavelet_features_size), dtype='f', chunks=(1, len(leads), self.wavelet_features_size))
        nodriftset = grp.create_dataset("drift_removed", (1, len(leads), self.window_size), maxshape=(None, len(leads), self.window_size),dtype='f',chunks=(1, len(leads), self.window_size))
        i = 0
        for recording_file, header_file in data:
        
            print(f"Iterating over {i +1} out of {num_recordings} files - {header_file}")
            # extract features of the recording, source will be ignored as it may be a redherring
            # Load header and recording.
            header = load_header(header_file)
            try:
                current_label= get_label(header)
            except Exception as e:
                print("Failed to load label, assigning 0")
                current_label = 0

            try:
                 age, sex_one_hot_encoding, source, signal_mean, signal_std, signal, fields = self.extract_features(recording_file)
            except Exception as e:
                print(f"Skipping {header_file} and associated recording  because of {e}")
                continue

            recording_features_record = np.concatenate((age, sex_one_hot_encoding, signal_mean, signal_std))
            print(recording_features_record.shape)

            weight_multiplier = 1
            if source is None:
                source = ""
            #if "sami" not in source.lower():
            #    weight_multiplier = 100
                    




            recording_full = self.load_and_equalize_recording(signal, fields, header_file, sampling_rate)
            if recording_full is None:
                print(f"Failed to load any data from {recording_file}, skipping")
                continue
            if recording_full.max() == 0 and recording_full.min() == 0:
                print("Skipping {recording_files[i]} as recording full seems to be none or empty")
                continue
            drift_removed_recording, recording, signals, infos, peaks, rates = self.preprocess_recording(recording_full, header,  leads_idxs=leads_idxs_dict[len(leads)])
            if signals is None or infos is None or peaks is None or rates is None:
                if signals is None:
                    print(f"Signals is none")
                if infos is None:
                    print("Infos is none")
                if peaks is None:
                    print("Peaks is None")
                if rates is None:
                    print("Rates is none")
                continue
            recording_raw, recording_drift_removed, rr_features, wavelet_features = self.one_file_training_data(recording, drift_removed_recording, signals, infos, rates, self.window_size,peaks, header_file, leads=leads)

            #recording_raw = np.repeat(recording_raw, weight_multiplier, axis=0)
            #recording_drift_removed = np.repeat(recording_drift_removed, weight_multiplier, axis=0)
            #rr_features = np.repeat(rr_features, weight_multiplier, axis=0)
            #wavelet_features = np.repeat(wavelet_features, weight_multiplier, axis=0)
            
            new_windows = recording_raw.shape[0]
            recording_features_repeated = np.repeat([recording_features_record], new_windows, axis=0)
            
            if new_windows == 0:
                logger.debug("New windows is 0! SKIPPING")
                continue
            label_pack = None
            if current_label:
                label_pack = np.ones((new_windows, 1), dtype=np.bool_)
                pos_signals += new_windows # * weight_multiplier)
            else:
                neg_signals += new_windows * weight_multiplier
                label_pack = np.zeros((new_windows, 1), dtype=np.bool_)
            dset.resize(dset.shape[0] + new_windows, axis=0)
            dset[-new_windows:] = recording_raw
            lset.resize(lset.shape[0] + new_windows, axis=0)
            lset[-new_windows:] = label_pack
            recording_features.resize(recording_features.shape[0] + new_windows, axis=0)
            recording_features[-new_windows:] = recording_features_repeated
            rrset.resize(rrset.shape[0] + new_windows, axis=0)
            rrset[-new_windows:] = rr_features
            waveset.resize(waveset.shape[0] + new_windows, axis=0)
            if wavelet_features.shape[0] != new_windows:
                waveset[-new_windows:] = wavelet_features[:-1]
            else:
                waveset[-new_windows:] = wavelet_features
            nodriftset.resize(nodriftset.shape[0] + new_windows, axis=0)
            nodriftset[-new_windows:] = recording_drift_removed
            print(f"Positive class counter after file: {pos_signals}")
            print(f"Negative class counter after file: {neg_signals}")
            i += 1
        print(f'Successfully created {group} dataset {filename}')
        return pos_signals, neg_signals


    def load_training_weights(self):
        data = []
        with open(self.training_weights_filename, 'r') as f:
            reader = csv.reader(f)
            for _, line in enumerate(reader):
                data.append(list(line))
            logger.debug(f"Loaded weights from CSV Reader: {data}")
        weights=torch.from_numpy(np.array(data, dtype=np.float32)).to(self.device)
        pos_weights = weights[0]
        neg_weights = weights[1]

        final_weight = neg_weights/pos_weights
        return final_weight

    def load_test_headers_and_recordings(self, fold, leads):
        test_filename = self.test_filename.format(leads,fold)
        with open(f"{test_filename}_header_recording_files.json", 'r') as f:
            loaded_dict = json.load(f)
            return (loaded_dict['header_files'], loaded_dict['recording_files'])


    #TODO zdefiniować mądrzejsze ogarnianie device
    def load_model(self, filename, alpha_config, beta_config, gamma_config, delta_config, epsilon_config, zeta_config, classes, leads, device):
        checkpoint = torch.load(filename, map_location=torch.device(device))
        #model = get_BlendMLP(alpha_config, beta_config, classes,device, leads=leads)
        model = get_MultibranchBeats(alpha_config, beta_config, gamma_config, delta_config, epsilon_config, zeta_config, classes,device, leads=leads)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.leads = checkpoint['leads']
        model.to(device)
        logger.info(f'Restored checkpoint from {filename}.') 
        return model

