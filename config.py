from datetime import datetime

class ARGS():
    def __init__(self):
        self.input = "/home/data/"
        self.output =  "/home/results/"
        self.target =  "data/h5_datasets/"
        self.gpu = 0
        self.model = "/home/models/"
        self.clean = False
        self.window_size=1500
        self.wavelet_features_size=759
        self.alpha_input_size=1500
        self.beta_input_size=759
        self.gamma_input_size=10668
        self.delta_input_size=256
        self.epsilon_input_size=1500
        self.zeta_input_size=1500
        self.name = f"Physionet_RUN_{datetime.today().strftime('%Y-%m-%d')}"
        self.debug_mode = False
        self.remove_baseline = False
        self.fold_to_process = ""
        self.network = "NBEATS"
        self.include_domain = False
        self.alpha_hidden=11
        self.alpha_layers=2
        self.beta_hidden=11
        self.beta_layers=2
        self.epochs=30
        self.early_stop=15
        self.fold = 1
        self.leads = "12"



