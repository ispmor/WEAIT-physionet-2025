from datetime import datetime

class ARGS():
    def __init__(self):
        self.input = "/home/data/"
        self.output =  "/home/results/"
        self.target =  "data/h5_datasets/"
        self.gpu = 2
        self.model = "/home/models/"
        self.clean = False
        self.window_size=1500
        self.wavelet_features_size=759
        self.alpha_input_size=1500
        self.beta_input_size=759
        self.gamma_input_size=256
        self.delta_input_size=10668
        self.epsilon_input_size=1500
        self.zeta_input_size=1500
        self.debug_mode = False
        self.remove_baseline = False
        self.fold_to_process = ""
        self.network = "LSTM"
        self.include_domain = False
        self.alpha_hidden=18
        self.alpha_layers=16
        self.beta_hidden=18
        self.beta_layers=16
        self.epochs=30
        self.early_stop=6
        self.fold = 1
        self.leads = "12"
        self.name = f"{self.network}_NO_DELTA_{self.leads}_{datetime.today().strftime('%Y-%m-%d')}"


