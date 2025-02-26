

class CustomLSTMPeephole_ALPHA(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_size,
                 num_layers,
                 seq_length,
                 device,
                 model_type='alpha',
                 classes=[]):
        super(CustomLSTMPeephole_ALPHA, self).__init__()

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

        print(f'| LSTM_PEEPHOLE_ALPHA')

        # self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.lstmpeephole_alpha1 = CustomLSTMPeephole(input_size=input_size, hidden_size=hidden_size)
        self.lstmpeephole_alpha2 = CustomLSTMPeephole(input_size=input_size, hidden_size=hidden_size)

        self.fc_1 = nn.Linear(hidden_size * 541, 128)  # hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer
        self.relu = nn.ReLU()

    def forward(self, alpha1_input, alpha2_input):
        h_0 = autograd.Variable(torch.zeros(alpha1_input.size(0), self.hidden_size,
                                            device=self.device))  # hidden state
        c_0 = autograd.Variable(torch.zeros(alpha1_input.size(0), self.hidden_size,
                                            device=self.device))  # internal state
        h_1 = autograd.Variable(torch.zeros(alpha2_input.size(0), self.hidden_size,
                                            device=self.device))  # hidden state
        c_1 = autograd.Variable(torch.zeros(alpha2_input.size(0), self.hidden_size,
                                            device=self.device))  # internal state

        output = []
        oa1, (h_0, c_0) = self.lstmpeephole_alpha1(alpha1_input, (h_0, c_0))  # lstm with input, hidden, and internal state
        oa2, (h_1, c_1) = self.lstmpeephole_alpha2(alpha2_input, (h_1, c_1))  # lstm with input, hidden, and internal state

        tmp = torch.hstack((oa1, oa2))
        tmp = torch.flatten(tmp, start_dim=1)

        out = self.fc_1(tmp)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out



class CustomLSTMPeephole_BETA(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_size,
                 num_layers,
                 seq_length,
                 device,
                 model_type='beta',
                 classes=[]):
        super(CustomLSTMPeephole_BETA, self).__init__()

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

        print(f'| LSTM_PEEPHOLE_BETA')

        self.linea_multiplier = input_size
        if input_size > 6:
            self.linea_multiplier = 6
        self.input_size = 1

        self.lstmpeephole_beta = CustomLSTMPeephole(input_size=self.input_size, hidden_size=self.hidden_size)

        self.fc = nn.Linear(
            (input_size * self.linea_multiplier + 363 * self.linea_multiplier + self.linea_multiplier) * self.hidden_size,
            num_classes)
        self.relu = nn.ReLU()

    def forward(self, beta_input):
        h_0 = autograd.Variable(torch.zeros(beta_input.size(0), self.hidden_size,
                                            device=self.device))  # hidden state

        c_0 = autograd.Variable(torch.zeros(beta_input.size(0), self.hidden_size,
                                            device=self.device))  # hidden state

        oa1, (h_0, c_0) = self.lstmpeephole_beta(beta_input, (h_0, c_0))  # lstm with input, hidden, and internal state

        out = torch.flatten(oa1, start_dim=1)
        out = self.relu(out)  # relu
        out = self.fc(out) # Final Output
        return out







class JitLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(JitLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))

        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.weight_ch_i = nn.Parameter(torch.Tensor(hidden_size))
        self.weight_ch_f = nn.Parameter(torch.Tensor(hidden_size))
        self.weight_ch_o = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameter()

    def reset_parameter(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        batch_size, sequence_size, input_size = input.size()
        hidden_seq = []
        hx, cx = state
        for t in range(sequence_size):
            inp = input[:, t, :]
            xh = (torch.mm(inp, self.weight_ih.t()) + self.bias_ih + torch.mm(hx, self.weight_hh.t()) + self.bias_hh)

            i, f, _c, o = xh.chunk(4, 1)

            i = torch.sigmoid(i + (self.weight_ch_i * cx))
            f = torch.sigmoid(f + (self.weight_ch_f * cx))
            _c = torch.tanh(_c)

            cy = (f * cx) + (i * _c)

            o = torch.sigmoid(o + (self.weight_ch_o * cy))
            hy = o * torch.tanh(cy)
            hidden_seq.append(hy.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (hx, cx)


class LSTMPeephole_ALPHA(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_size,
                 num_layers,
                 seq_length,
                 device,
                 model_type='alpha',
                 classes=[]):
        super(LSTMPeephole_ALPHA, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.model_type = model_type
        self.classes = classes
        self.device=device
        self.sigmoid = nn.Sigmoid()
        self.when_bidirectional = 1  # if bidirectional = True, then it has to be equal to 2

        print(f'| LSTM_PEEPHOLE_ALPHA')

        # self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.lstmpeephole_alpha1 = JitLSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.lstmpeephole_alpha2 = JitLSTMCell(input_size=input_size, hidden_size=hidden_size)

        self.fc_1 = nn.Linear(hidden_size * 541, 128)  # hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer
        self.relu = nn.ReLU()

    def forward(self, alpha1_input, alpha2_input):
        h_0 = autograd.Variable(torch.zeros(alpha1_input.size(0), self.hidden_size,
                                            device=self.device))  # hidden state
        c_0 = autograd.Variable(torch.zeros(alpha1_input.size(0), self.hidden_size,
                                            device=self.device))  # internal state
        h_1 = autograd.Variable(torch.zeros(alpha2_input.size(0), self.hidden_size,
                                            device=self.device))  # hidden state
        c_1 = autograd.Variable(torch.zeros(alpha2_input.size(0), self.hidden_size,
                                            device=self.device))  # internal state


        oa1, _ = self.lstmpeephole_alpha1(alpha1_input,
                                                   (h_0, c_0))  # lstm with input, hidden, and internal state
        oa2, _ = self.lstmpeephole_alpha2(alpha2_input,
                                                   (h_1, c_1))  # lstm with input, hidden, and internal state

        tmp = torch.hstack((oa1, oa2))
        tmp = torch.flatten(tmp, start_dim=1)

        out = self.fc_1(tmp)  # first Dense
        del tmp, h_0, c_0, h_1, c_1
        out = self.relu(out)  # relu
        out = self.fc(out) # Final Output
        return out


class LSTMPeephole_BETA(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_size,
                 num_layers,
                 seq_length,
                 device,
                 model_type='beta',
                 classes=[]):
        super(LSTMPeephole_BETA, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.model_type = model_type
        self.classes = classes
        self.sigmoid = nn.Sigmoid()
        self.device=device
        self.when_bidirectional = 1  # if bidirectional = True, then it has to be equal to 2

        print(f'| LSTM_PEEPHOLE_BETA')

        self.linea_multiplier = input_size
        if input_size > 6:
            self.linea_multiplier = 6
        # self.hidden_size=1
        # self.num_layers=1
        self.input_size = 1

        # self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.lstmpeephole_beta = JitLSTMCell(input_size=self.input_size, hidden_size=hidden_size)

        self.fc = nn.Linear(
            (input_size * self.linea_multiplier + 363 * self.linea_multiplier + self.linea_multiplier) * hidden_size,
            num_classes)
        self.relu = nn.ReLU()

    def forward(self, beta_input):
        h_0 = autograd.Variable(torch.zeros(beta_input.size(0), self.hidden_size,
                                            device=self.device))  # hidden state

        c_0 = autograd.Variable(torch.zeros(beta_input.size(0), self.hidden_size,
                                            device=self.device))  # hidden state


        oa1, _  = self.lstmpeephole_beta(beta_input,
                                                 (h_0, c_0))  # lstm with input, hidden, and internal state

        out = torch.flatten(oa1, start_dim=1)
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out


class CustomLSTMPeephole(nn.Module):
    def __init__(self, input_size, hidden_size, peephole=True):
        super().__init__()
        self.input_sz = input_size
        self.hidden_size = hidden_size
        self.peephole = peephole
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x,
                init_states):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        h_t, c_t = init_states

        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication

            if self.peephole:
                gates = (torch.mm(x_t, self.U) + torch.mm(c_t, self.W) + self.bias)
            else:
                gates = x_t @ self.U + h_t @ self.W + self.bias
                g_t = torch.tanh(gates[:, HS * 2:HS * 3])

            i_t, f_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.sigmoid(gates[:, HS * 3:]),  # output
            )

            if self.peephole:
                c_t = f_t * c_t + i_t * torch.sigmoid(x_t @ self.U + self.bias)[:, HS * 2:HS * 3]
                h_t = torch.tanh(o_t * c_t)
            else:
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (h_t, c_t)


class GRU_ECG_ALPHA(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_size,
                 num_layers,
                 seq_length,
                 device,
                 model_type='alpha',
                 classes=[]):
        super(GRU_ECG_ALPHA, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.model_type = model_type
        self.classes = classes
        self.device=device
        self.sigmoid = nn.Sigmoid()
        self.when_bidirectional = 1  # if bidirectional = True, then it has to be equal to 2
        print(f'| GRU_ECG_ALPHA')

        # self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.gru_alpha1 = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                                 num_layers=num_layers, batch_first=True, bidirectional=False)
        self.gru_alpha2 = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                                 num_layers=num_layers, batch_first=True, bidirectional=False)

        self.fc_1 = nn.Linear(hidden_size * 555, 128)  # hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer
        self.relu = nn.ReLU()

    def forward(self, alpha1_input, alpha2_input):
        h_0 = autograd.Variable(torch.zeros(self.num_layers * self.when_bidirectional, alpha1_input.size(0), self.hidden_size,
                                            device=self.device))  # hidden state
        h_1 = autograd.Variable(torch.zeros(self.num_layers * self.when_bidirectional, alpha1_input.size(0), self.hidden_size,
                                            device=self.device))  # hidden state

        output_alpha1, hn_alpha1 = self.gru_alpha1(alpha1_input, h_0)  # lstm with input, hidden, and internal state
        output_alpha2, hn_alpha2 = self.gru_alpha2(alpha2_input, h_1)  # lstm with input, hidden, and internal state
        tmp = torch.hstack((output_alpha1, output_alpha2))
        tmp = torch.flatten(tmp, start_dim=1)

        out = self.fc_1(tmp)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out


class GRU_ECG_BETA(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_size,
                 num_layers,
                 seq_length,
                 device,
                 model_type='alpha',
                 classes=[]):
        super(GRU_ECG_BETA, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.model_type = model_type
        self.classes = classes
        self.device=device
        self.sigmoid = nn.Sigmoid()
        self.when_bidirectional = 1  # if bidirectional = True, then it has to be equal to 2
        print(f'| GRU_ECG_BETA')

        self.linea_multiplier = input_size
        if input_size > 6:
            self.linea_multiplier = 6
        self.input_size = 1
        self.gru_beta = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                               num_layers=self.num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(
            (input_size * self.linea_multiplier + 370 * self.linea_multiplier + self.linea_multiplier) * hidden_size,
            num_classes)
        self.relu = nn.ReLU()

    def forward(self, beta_input):
        h_0 = autograd.Variable(
            torch.zeros(self.num_layers * self.when_bidirectional, beta_input.size(0), self.hidden_size,
                        device=self.device))  # hidden state
        output_beta, hn_beta = self.gru_beta(beta_input, h_0)

        out = torch.flatten(output_beta, start_dim=1)
        out = self.relu(out)  # relua
        out = self.fc(out)  # Final Output
        return out

