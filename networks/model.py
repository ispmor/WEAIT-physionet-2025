
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torch import jit
import logging
import math


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
                 input_features_size=363):
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

        if model_type == 'alpha':
            linear_input_size = input_features_size * input_size
        else:
            self.linea_multiplier = input_size
            if input_size > 6:
                self.linea_multiplier = 6
            linear_input_size = input_size * self.linea_multiplier + input_features_size * self.linea_multiplier + self.linea_multiplier
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
            block_init = NBeatsNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(self.hidden_layer_units, self.thetas_dim[stack_id], self.input_features_size * self.input_size,
                                   self.target_size, classes=len(self.classes))
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
    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, share_thetas=False, classes=16):
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
        logger.debug(f"NBeats Block forward - FC1 output shape: {x.shape}")
        x = F.relu(self.fc2(x))
        logger.debug(f"NBeats Block forward - FC2 output  shape: {x.shape}")
        x = F.relu(self.fc3(x))
        logger.debug(f"NBeats Block forward - FC3 output  shape: {x.shape}")
        x = F.relu(self.fc4(x))
        logger.debug(f"NBeats Block forward - FC4 output  shape: {x.shape}")
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class GenericBlock(Block):

    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, classes=16):
        super(GenericBlock, self).__init__(units, thetas_dim, backcast_length, forecast_length, classes=classes)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, backcast_length)  # forecast_length)

    def forward(self, x):
        logger.debug(f"NBeats Generic Block forward. Input shape: {x.shape}")
        x = super(GenericBlock, self).forward(x)

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))  # tutaj masz thetas_dim rozmiar

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast


class Nbeats_alpha(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_size,
                 num_layers,
                 seq_length,
                 device,
                 classes=[],
                 model_type='alpha',
                 input_features_size_a1=350,
                 input_features_size_a2=185):
        super(Nbeats_alpha, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.model_type = model_type
        self.device=device
        self.classes = classes
        self.relu = nn.ReLU()

        self.nbeats_alpha1 = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK],
                                       nb_blocks_per_stack=self.num_layers,
                                       target_size=num_classes,
                                       input_size=input_size,
                                       thetas_dims=(32, 32),
                                       device=device,
                                       classes=self.classes,
                                       hidden_layer_units=self.hidden_size,
                                       input_features_size=input_features_size_a1)

        self.nbeats_alpha2 = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK],
                                       nb_blocks_per_stack=self.num_layers,
                                       target_size=num_classes,
                                       input_size=input_size,
                                       thetas_dims=(32, 32),
                                       device=device,
                                       classes=self.classes,
                                       hidden_layer_units=hidden_size,
                                       input_features_size=input_features_size_a2)

        self.fc_1 = nn.Linear(self.input_size * (input_features_size_a1 + input_features_size_a2), 128)  # hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

    def forward(self, alpha1_input, alpha2_input):
        _, output_alpha1 = self.nbeats_alpha1(alpha1_input)  # lstm with input, hidden, and internal state
        _, output_alpha2 = self.nbeats_alpha2(alpha2_input)  # lstm with input, hidden, and internal state
        logger.debug(f"ALPHA 1 Output Shape: {output_alpha1.shape}\nALPHA 2 Output shape: {output_alpha2.shape}")

        tmp = torch.hstack((output_alpha1, output_alpha2))
        tmp = torch.flatten(tmp, start_dim=1)

        out = self.fc_1(tmp)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out


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
                 input_features_size_b=360):
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
                                     input_features_size=input_features_size_b)

        self.fc = nn.Linear( input_features_size_b * self.input_size,
                            num_classes)  # hidden_size, 128)  # fully connected 1# fully connected last layer
        logger.debug(f"{self}")


    def forward(self, beta_input):
        logger.debug(f"NBeats_beta INPUT shape: {beta_input.shape}")
        beta_flattened = torch.flatten(beta_input, start_dim=1)
        logger.debug(f"NBeats_beta INPUT FLATTENED shape: {beta_flattened.shape}")
        _, output_beta = self.nbeats_beta(beta_flattened)  # lstm with input, hidden, and internal state
        logger.debug(f"Nbeats_beta OUTPUT shape: {output_beta.shape}")
        tmp = torch.squeeze(output_beta)
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
            # self.hidden_size=1
            # self.num_layers=1
            #self.input_size = 1
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
            output_beta = self.dropoutLstmA(output_beta)
            out = torch.flatten(output_beta, start_dim=1)
            out = self.relu(out)  # relu
            out = self.fc(out)  # Final Output
            out = self.dropoutFC(out)
        return out




class BlendMLP(nn.Module):
    def __init__(self, modelA, modelB, classes):
        super(BlendMLP, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classes = classes
        self.linear = nn.Linear(2 * len(classes) + 20, len(classes))

    def forward(self, alpha1_input, alpha2_input, beta_input, rr):
        x1 = self.modelA(alpha1_input, alpha2_input)
        x2 = self.modelB(beta_input)

        if x1.shape == x2.shape:
            out = F.relu(torch.cat((x1, x2), dim=1))
            out_with_rr = torch.cat((out, rr), dim=1)
            out = self.linear(out_with_rr)
            return out
        else:
            return x1
        return x2

class MultibranchBeats(nn.Module):
    def __init__(self, modelA, modelB, modelC, modelD, modelE, modelF, classes):
        super(MultibranchBeats, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.modelD = modelD
        self.modelE = modelE
        self.modelF = modelF
        self.classes = classes
        self.linear = nn.Linear( 6 * len(classes), len(classes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, alpha_input, beta_input, gamma_input, delta_input, epsilon_input, zeta_input):
        logger.debug(f"Alpha input shape: {alpha_input.shape}\nBeta input shape: {beta_input.shape}\nGamma input shape: {gamma_input.shape}\nDelta input shape: {delta_input.shape}")

        outA = self.modelA(alpha_input)
        outB = self.modelB(beta_input)
        outC = self.modelC(gamma_input)
        outD = self.modelD(delta_input)
        outE = self.modelE(epsilon_input)
        outF = self.modelF(zeta_input)

        out_concat = F.relu(torch.cat((outA, outB, outC, outD, outE, outF), dim=1))
        out = self.linear(out_concat)
        return out


def get_single_network(network, hs, layers, leads, selected_classes, single_peak_length,a1_in, a2_in, b_in, as_branch, device):
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
        if as_branch == "alpha":
            return Nbeats_alpha(input_size=leads,
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
            return Nbeats_beta(input_size=leads,
                            num_classes=len(selected_classes),
                            hidden_size=hs,
                            seq_length=single_peak_length,
                            device=device,
                            model_type=as_branch,
                            classes=selected_classes,
                            num_layers=layers,
                            input_features_size_b=b_in)


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



def get_BlendMLP(alpha_config: BranchConfig, beta_config: BranchConfig, classes: list, device, leads: list) -> BlendMLP:
    alpha_branch = get_single_network(alpha_config.network_name, alpha_config.hidden_size, alpha_config.layers, leads, classes, alpha_config.single_peak_length, alpha_config.a1_input_size, alpha_config.a2_input_size, None, "alpha", device)
    beta_branch = get_single_network(beta_config.network_name, beta_config.hidden_size, beta_config.layers, leads, classes, beta_config.single_peak_length, None, None, beta_config.beta_input_size, "beta", device)

    return BlendMLP(alpha_branch, beta_branch, classes)



def get_MultibranchBeats(alpha_config: BranchConfig, beta_config: BranchConfig, gamma_config: BranchConfig, delta_config: BranchConfig, epsilon_config: BranchConfig, zeta_config: BranchConfig, classes: list, device, leads: list) -> MultibranchBeats:
    alpha_branch = get_single_network(alpha_config.network_name, alpha_config.hidden_size, alpha_config.layers, len(leads), classes, alpha_config.single_peak_length, None, None, alpha_config.beta_input_size, "beta", device)
    beta_branch = get_single_network(beta_config.network_name, beta_config.hidden_size, beta_config.layers, len(leads), classes, beta_config.single_peak_length, None, None, beta_config.beta_input_size, "beta", device)
    gamma_branch = get_single_network(gamma_config.network_name, gamma_config.hidden_size, gamma_config.layers, gamma_config.channels, classes, gamma_config.single_peak_length, None, None, gamma_config.beta_input_size, "beta", device)
    delta_branch = get_single_network(delta_config.network_name, delta_config.hidden_size, delta_config.layers, len(leads), classes, delta_config.single_peak_length, None, None, delta_config.beta_input_size, "beta", device)
    epsilon_branch = get_single_network(epsilon_config.network_name, epsilon_config.hidden_size, epsilon_config.layers, len(leads), classes, epsilon_config.single_peak_length, None, None, epsilon_config.beta_input_size, "beta", device)
    zeta_branch = get_single_network(zeta_config.network_name, zeta_config.hidden_size, zeta_config.layers, len(leads), classes, zeta_config.single_peak_length, None, None, zeta_config.beta_input_size, "beta", device)

    return MultibranchBeats(alpha_branch, beta_branch, gamma_branch, delta_branch, epsilon_branch, zeta_branch, classes)

