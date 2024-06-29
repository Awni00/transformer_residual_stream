import torch
from torch import nn

class ResidualGate(nn.Module):
    def __init__(self,
            d_model: int,
            gate_application: str = 'reset-update', # 'reset-update' or 'reset' or 'update' or 'combined' or 'none'
            gate_compute: str = 'linear-bias', # 'linear-bias' or 'linear' or 'bias'
            gate_activation: str = 'sigmoid', # 'sigmoid' or 'tanh' or 'none
            ):
        super(ResidualGate, self).__init__()

        self.d_model = d_model
        self.gate_application = gate_application
        self.gate_compute = gate_compute
        self.gate_activation = gate_activation
        self.gate_activation_fn = nn.Sigmoid() if gate_activation == 'sigmoid' else nn.Tanh() if gate_activation == 'tanh' else None


        if gate_compute in ['linear-bias', 'linear']:
            bias = (gate_compute == 'linear-bias')
            if gate_application == 'reset-update':
                self.update_gate_linear = nn.Linear(d_model, d_model, bias=bias)
                self.reset_gate_linear = nn.Linear(d_model, d_model, bias=bias)
            elif gate_application == 'reset' or gate_application == 'combined':
                self.reset_gate_linear = nn.Linear(d_model, d_model, bias=bias)
            elif gate_application == 'update':
                self.update_gate_linear = nn.Linear(d_model, d_model, bias=bias)
            elif gate_application == 'none':
                pass
            else:
                raise ValueError(f'Unknown gate_application: {gate_application}')

        elif gate_compute == 'bias':
            if gate_application == 'reset-update':
                self.update_gate_bias = nn.Parameter(torch.zeros(d_model))
                self.reset_gate_bias = nn.Parameter(torch.zeros(d_model))
            elif gate_application == 'reset' or gate_application == 'combined':
                self.reset_gate_bias = nn.Parameter(torch.zeros(d_model))
                # in the combined case, the reset gate is used to compute g*x + (1-g)*y
            elif gate_application == 'update':
                self.update_gate_bias = nn.Parameter(torch.zeros(d_model))
            elif gate_application == 'none':
                pass
            else:
                raise ValueError(f'Unknown gate_application: {gate_application}')
        elif gate_application != 'none':
            raise ValueError(f'Unknown gate_compute: {gate_compute}')

    # TODO: bias initialization (non-zero)

    def _compute_update_gate(self, x):
        if self.gate_compute in ('linear', 'linear-bias'):
            update_gate = self.gate_activation_fn(self.update_gate_linear(x))
        elif self.gate_compute == 'bias':
            update_gate = self.gate_activation_fn(torch.zeros_like(x) + self.update_gate_bias)
        else:
            raise ValueError(f'Unknown gate_compute: {self.gate_compute}')

        return update_gate

    def _compute_reset_gate(self, x):
        if self.gate_compute in ('linear', 'linear-bias'):
            reset_gate = self.gate_activation_fn(self.reset_gate_linear(x))
        elif self.gate_compute == 'bias':
            reset_gate = self.gate_activation_fn(torch.zeros_like(x) + self.reset_gate_bias)
        else:
            raise ValueError(f'Unknown gate_compute: {self.gate_compute}')

        return reset_gate


    def forward(self, x, y):
        if self.gate_application == 'none':
            z = x + y
        elif self.gate_application == 'update':
            update_gate = self._compute_update_gate(x)
            z = update_gate * x + y
        elif self.gate_application == 'reset':
            reset_gate = self._compute_reset_gate(x)
            z = reset_gate * x + y
        elif self.gate_application == 'reset-update':
            update_gate = self._compute_update_gate(x)
            reset_gate = self._compute_reset_gate(x)
            z = reset_gate * x + update_gate * y
        elif self.gate_application == 'combined':
            gate = self._compute_reset_gate(x)
            z = gate * x + (1 - gate) * y
        else:
            raise ValueError(f'Unknown gate_application: {self.gate_application}')

        return z

# TODO: all the gating mechanisms above are x-dependent but not y-dependent
# in LSTM, for e.g., gates are both x and y deppendent. i.e., gate = sigmoid(Wx + Uy + b)