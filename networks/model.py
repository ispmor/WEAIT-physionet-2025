
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
import logging


logger = logging.getLogger(__name__)


class NBeatsNet(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(self,
                 stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
                 nb_blocks_per_stack=1,
                 target_size=5,
                 input_size=10,
                 thetas_dims=(4, 8),
                 share_weights_in_stack=False,
                 hidden_layer_units=17,
                 device=None,
                 classes=[],
                 model_type='alpha',
                 input_features_size=363,
                 dropout_rate=0.0):
        super(NBeatsNet, self).__init__()
        self.classes = classes
        self.leads = []
        self.target_size = target_size
        self.input_size = input_size
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.input_features_size=input_features_size
        self.stack_types = stack_types
        self.stacks = []
        self.thetas_dim = thetas_dims
        self.device = device
        self.parameters = []
        self.dropout_rate = dropout_rate

        if model_type == 'alpha':
            linear_input_size = input_features_size * input_size
        else:
            self.linea_multiplier = input_size
            if input_size > 6:
                self.linea_multiplier = 6
        self.fc_linear = nn.Linear(input_features_size * len(classes), len(classes))

        print(f'| N-Beats, device={self.device}')

        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = GenericBlock(self.hidden_layer_units, self.thetas_dim[stack_id], self.input_size, self.target_size, classes=len(self.classes), dropout_rate=self.dropout_rate)
                self.parameters.extend(block.parameters())
            print(f'     | -- {block}')
            blocks.append(block)
        return blocks

    @staticmethod
    def select_block(block_type):
        return GenericBlock

    def forward(self, backcast):
        forecast = torch.zeros(size=backcast.shape, device=self.device)
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast - b
                forecast = forecast + f

        return backcast, forecast


def linspace(backcast_length, forecast_length):
    lin_space = np.linspace(-backcast_length, forecast_length, backcast_length + forecast_length)
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls


class Block(nn.Module):
    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, share_thetas=False, classes=1, dropout_rate: float=0.0):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        self.dropout_rate = dropout_rate
        if dropout_rate != 0.0:
            self.dropout_fc1 = nn.Dropout(dropout_rate)
            self.dropout_fc2 = nn.Dropout(dropout_rate)
            self.dropout_fc3 = nn.Dropout(dropout_rate)
            self.dropout_fc4 = nn.Dropout(dropout_rate)
            self.dropout_theta_b_fc = nn.Dropout(dropout_rate)
            self.dropout_theta_f_fc = nn.Dropout(dropout_rate)
            logger.debug("Created droputs")
        self.backcast_linspace, self.forecast_linspace = linspace(backcast_length, forecast_length)
        self.classes = classes

        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim)
            self.theta_f_fc = nn.Linear(units, thetas_dim)
            

    def forward(self, x):
        logger.debug(f"NBeats Block forward - INPUT  shape: {x.shape}")
        x = F.relu(self.fc1(x))
        if self.dropout_rate != 0.0:
            logger.debug(f"Dropout rate at FC1 in nbeats is: {self.dropout_rate}")
            x = self.dropout_fc1(x)
        logger.debug(f"NBeats Block forward - FC1 output shape: {x.shape}")
        x = F.relu(self.fc2(x))
        if self.dropout_rate != 0.0:
            x = self.dropout_fc2(x)
        logger.debug(f"NBeats Block forward - FC2 output  shape: {x.shape}")
        x = F.relu(self.fc3(x))
        if self.dropout_rate != 0.0:
            x = self.dropout_fc3(x)
        logger.debug(f"NBeats Block forward - FC3 output  shape: {x.shape}")
        x = F.relu(self.fc4(x))
        if self.dropout_rate != 0.0:
            x = self.dropout_fc4(x)
        logger.debug(f"NBeats Block forward - FC4 output  shape: {x.shape}")
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class GenericBlock(Block):

    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, classes=16, dropout_rate=0.0):
        super(GenericBlock, self).__init__(units, thetas_dim, backcast_length, forecast_length, classes=classes, dropout_rate=dropout_rate)
        logger.debug(f"At generic block creation droput_rate: {dropout_rate}")
        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, backcast_length)  # forecast_length)

    def forward(self, x):
        logger.debug(f"NBeats Generic Block forward. Input shape: {x.shape}")
        x = super(GenericBlock, self).forward(x)

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))  # tutaj masz thetas_dim rozmiar
        if self.dropout_rate != 0.0:
            theta_b = self.dropout_theta_b_fc(theta_b)
            theta_f = self.dropout_theta_f_fc(theta_f)


        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast




class Nbeats_beta(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_size,
                 num_layers,
                 seq_length,
                 device,
                 classes=[],
                 model_type='beta',
                 input_features_size_b=360,
                 dropout_rate=0.0):
        super(Nbeats_beta, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.model_type = model_type
        self.classes = classes
        self.device = device
        self.input_size = input_size
        self.input_features_size = input_features_size_b
        self.relu = nn.ReLU()

        self.linea_multiplier = input_size
        if input_size > 6:
            self.linea_multiplier = 6
        # self.hidden_size = 1
        # self.num_layers = 3

        self.nbeats_beta = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK],
                                     nb_blocks_per_stack=self.num_layers,
                                     target_size=num_classes,
                                     input_size=self.input_size,
                                     thetas_dims=(32, 32),
                                     device=self.device,
                                     classes=self.classes,
                                     hidden_layer_units=self.hidden_size,
                                     input_features_size=input_features_size_b,
                                     dropout_rate=dropout_rate)

        self.fc = nn.Linear( input_features_size_b * self.input_size,
                            num_classes)  # hidden_size, 128)  # fully connected 1# fully connected last layer
        self.dropoutNBEATS = nn.Dropout(dropout_rate)
        logger.debug(f"{self}")


    def forward(self, beta_input):
        logger.debug(f"NBeats_beta INPUT shape: {beta_input.shape}")
        #beta_flattened = torch.flatten(beta_input, start_dim=1)
        #logger.debug(f"NBeats_beta INPUT FLATTENED shape: {beta_flattened.shape}")
        _, output_beta = self.nbeats_beta(beta_input)  # lstm with input, hidden, and internal state
        logger.debug(f"Nbeats_beta OUTPUT shape: {output_beta.shape}")
        output_beta = self.dropoutNBEATS(output_beta)
        logger.debug(f"Nbeats_beta DROPOUT OUTPUT shape: {output_beta.shape}")
        tmp = torch.flatten(output_beta, start_dim=1)
        out = self.relu(tmp)  # relu
        out = self.fc(out)  # Final Output
        return out


class LSTM_ECG(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_size,
                 num_layers,
                 seq_length,
                 device,
                 model_type='alpha',
                 classes=[],
                 input_features_size_a1=350,
                 input_features_size_a2=185,
                 input_features_size_b=360):
        super(LSTM_ECG, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.model_type = model_type
        self.classes = classes
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.when_bidirectional = 1  # if bidirectional = True, then it has to be equal to 2
        self.dropoutLstmA = nn.Dropout(0.2)
        self.dropoutFC = nn.Dropout(0.2)

        print(f'| LSTM_ECG')

        self.lstm_alpha1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                   num_layers=num_layers, batch_first=True, bidirectional=False)
        if model_type == 'alpha':
            self.lstm_alpha2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                       num_layers=num_layers, batch_first=True, bidirectional=False)

            self.fc_1 = nn.Linear(hidden_size * (input_features_size_a1+input_features_size_a2), 128)  # hidden_size, 128)  # fully connected 1
            self.fc = nn.Linear(128, num_classes)  # fully connected last layer
        else:
            self.linea_multiplier = input_size
            if input_size > 6:
                self.linea_multiplier = 6
            self.lstm_alpha1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                                       num_layers=self.num_layers, batch_first=True, bidirectional=False)
            #self.fc = nn.Linear((input_size + input_features_size_b + 1) * self.linea_multiplier * self.hidden_size, num_classes)
            self.fc = nn.Linear( input_features_size_b * self.hidden_size, num_classes)

        self.relu = nn.ReLU()

    def forward(self, alpha1_input=None, alpha2_input=None):
        if self.model_type == 'alpha':
            h_0 = autograd.Variable(
                torch.zeros(self.num_layers * self.when_bidirectional, alpha1_input.size(0), self.hidden_size,
                            device=self.device))  # hidden state
            c_0 = autograd.Variable(
                torch.zeros(self.num_layers * self.when_bidirectional, alpha1_input.size(0), self.hidden_size,
                            device=self.device))  # internal state
            h_1 = autograd.Variable(
                torch.zeros(self.num_layers * self.when_bidirectional, alpha1_input.size(0), self.hidden_size,
                            device=self.device))  # hidden state
            c_1 = autograd.Variable(
                torch.zeros(self.num_layers * self.when_bidirectional, alpha1_input.size(0), self.hidden_size,
                            device=self.device))  # internal state

            output_alpha1, (hn_alpha1, cn) = self.lstm_alpha1(alpha1_input,
                                                              (h_0, c_0))  # lstm with input, hidden, and internal state
            output_alpha2, (hn_alpha2, cn) = self.lstm_alpha2(alpha2_input,
                                                              (h_1, c_1))  # lstm with input, hidden, and internal state
            tmp = torch.hstack((output_alpha1, output_alpha2))
            tmp = torch.flatten(tmp, start_dim=1)

            out = self.fc_1(tmp)  # first Dense
            out = self.relu(out)  # relu
            out = self.fc(out)  # Final Output
            return out
        else:
            alpha2_input=alpha1_input #as we pass only one argument which is pca vector in fact
            h_0 = autograd.Variable(
                torch.zeros(self.num_layers * self.when_bidirectional, alpha2_input.size(0), self.hidden_size,
                            device=self.device))  # hidden state
            c_0 = autograd.Variable(
                torch.zeros(self.num_layers * self.when_bidirectional, alpha2_input.size(0), self.hidden_size,
                            device=self.device))  # internal state

            output_beta, (hn_beta, cn) = self.lstm_alpha1(alpha2_input, (h_0, c_0))
            logger.debug(f"LSTM_beta OUTPUT shape: {output_beta.shape}")
            output_beta = self.dropoutLstmA(output_beta)
            logger.debug(f"LSTM_beta DROPOUT OUTPUT shape: {output_beta.shape}")
            out = torch.flatten(output_beta, start_dim=1)
            out = self.relu(out)  # relu
            out = self.fc(out)  # Final Output
            out = self.dropoutFC(out)
        return out




class MultibranchBeats(nn.Module):
    def __init__(self, modelA, modelB, modelC, modelD, modelE, classes):
        super(MultibranchBeats, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.modelD = modelD
        self.modelE = modelE
        self.modelF = nn.Linear(28, len(classes))
        self.classes = classes
        self.linear = nn.Linear( 6 * len(classes), len(classes)) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, alpha_input, beta_input, gamma_input, delta_input, epsilon_input, recording_features):
        logger.debug(f"Alpha input shape: {alpha_input.shape}\nBeta input shape: {beta_input.shape}\nGamma input shape: {gamma_input.shape}\nDelta input shape: {delta_input.shape}")
        logger.debug(f"Dataset label: {recording_features.shape}")

        #out = self.modelA(alpha_input)
        #out = self.modelB(beta_input)
        #out = self.modelC(gamma_input)
        out = self.modelD(delta_input)
        #joutE = self.modelE(epsilon_input)
        #outF = self.modelF(recording_features)
        #logger.debug(f"Alpha output shape: {outA.shape}\nBeta output shape: {outB.shape}\nGamma output shape: {outC.shape}\nDelta output shape: {outD.shape}, Epsilon output shape: {outE.shape}, dataset label shape: {recording_features.shape}")


        #out_concat = F.relu(torch.cat((outA, outB, outC, outD, outE, outF), dim=1))
        #out = self.linear(out_concat)
        out = F.relu(out)
        return out


def get_single_network(network, hs, layers, leads, selected_classes, single_peak_length,a1_in, a2_in, b_in, as_branch, device, dropout_rate=0.0):
    torch.manual_seed(17)

    if network == "LSTM":
        if as_branch == "alpha":
            return LSTM_ECG(input_size=leads,
                num_classes=len(selected_classes),
                hidden_size=hs,
                num_layers=layers,
                seq_length=single_peak_length,
                device=device,
                model_type=as_branch,
                classes=selected_classes,
                input_features_size_a1=a1_in,
                input_features_size_a2=a2_in)
        else:
            return LSTM_ECG(input_size=leads,
                num_classes=len(selected_classes),
                hidden_size=hs,
                num_layers=layers,
                seq_length=single_peak_length,
                device=device,
                model_type=as_branch,
                classes=selected_classes,
                input_features_size_b=b_in)


    if network == "NBEATS":
        return Nbeats_beta(input_size=leads,
                            num_classes=len(selected_classes),
                            hidden_size=hs,
                            seq_length=single_peak_length,
                            device=device,
                            model_type=as_branch,
                            classes=selected_classes,
                            num_layers=layers,
                            input_features_size_b=b_in,
                           dropout_rate=dropout_rate)


class BranchConfig:
    network_name = ""
    single_peak_length = -1
    hidden_size = -1 
    layers = -1
    
    def __init__(self,network_name, hidden_size, layers, single_peak_length, a1_input_size=None, a2_input_size=None, beta_input_size=None, channels=None) -> None:
        self.network_name = network_name
        self.single_peak_length=single_peak_length
        self.hidden_size=hidden_size
        self.layers=layers
        self.a1_input_size=a1_input_size
        self.a2_input_size=a2_input_size
        self.beta_input_size=beta_input_size
        self.channels = channels




def get_MultibranchBeats(alpha_config: BranchConfig, beta_config: BranchConfig, gamma_config: BranchConfig, delta_config: BranchConfig, epsilon_config: BranchConfig, classes: list, device, leads) -> MultibranchBeats:
    dropout_rate = 0.4
    alpha_branch = None#get_single_network(alpha_config.network_name, alpha_config.hidden_size, alpha_config.layers, len(leads), classes, alpha_config.single_peak_length, None, None, alpha_config.beta_input_size, "beta", device, dropout_rate)
    beta_branch = None#get_single_network(beta_config.network_name, beta_config.hidden_size, beta_config.layers, len(leads), classes, beta_config.single_peak_length, None, None, beta_config.beta_input_size, "beta", device, dropout_rate)
    gamma_branch = None#get_single_network(gamma_config.network_name, gamma_config.hidden_size, gamma_config.layers, len(leads), classes, gamma_config.single_peak_length, None, None, gamma_config.beta_input_size, "beta", device, dropout_rate)
    delta_branch = get_single_network(delta_config.network_name, delta_config.hidden_size, delta_config.layers, delta_config.channels, classes, delta_config.single_peak_length, None, None, delta_config.beta_input_size, "beta", device, dropout_rate)
    epsilon_branch = None#get_single_network(epsilon_config.network_name, epsilon_config.hidden_size, epsilon_config.layers, len(leads), classes, epsilon_config.single_peak_length, None, None, epsilon_config.beta_input_size, "beta", device, dropout_rate)

    return MultibranchBeats(alpha_branch, beta_branch, gamma_branch, delta_branch, epsilon_branch, classes)

