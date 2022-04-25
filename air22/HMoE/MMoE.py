"""
Multi-gate Mixture-of-Experts model implementation (PyTorch).
Written by Zhichen Zhao
"""
import torch
import torch.nn as nn

class MMoE(torch.nn.Module):# if you are not using pytorch lightning, you can also use 'Module'
    def __init__(self, input_size, units, num_experts, num_tasks, device, feature_vocabulary, embedding_size=5, use_expert_bias=False, use_gate_bias=False, expert_activation=None):
        super(MMoE, self).__init__()

        self.expert_kernels = torch.nn.Parameter(torch.randn(input_size, units, num_experts, device=device), requires_grad=True)
        self.gate_kernels = torch.nn.ParameterList([nn.Parameter(torch.randn(input_size, num_experts, device=device), requires_grad=True) for i in range(num_tasks)])

        self.expert_kernels_bias = torch.nn.Parameter(torch.randn(units, num_experts, device=device), requires_grad=True)
        self.gate_kernels_bias = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(num_experts, device=device), requires_grad=True) for i in range(num_tasks)])

        self.feature_vocabulary = feature_vocabulary
        self.feature_names = sorted(list(feature_vocabulary.keys()))
        self.embedding_size = embedding_size

        self.device = device
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_activation = expert_activation


    def forward(self, feature_embedding):
        '''
        x: input, (batch_size, input_size)
        expert_kernels: (input_size, units, num_experts)
        expert_kernels_bias: (units, num_experts)
        gate_kernels: (input_size, num_experts)
        gate_kernels_bias: (num_experts)
        final_outputs: output, a list len() == num_tasks, each element has shape of (batch_size, units)
        '''

        gate_outputs = []
        final_outputs = []

        expert_outputs = torch.einsum("ab,bcd->acd", (feature_embedding, self.expert_kernels))
        if self.use_expert_bias:
            expert_outputs += self.expert_kernels_bias

        if self.expert_activation is not None:
            expert_outputs = self.expert_activation(expert_outputs)

        for index, gate_kernel in enumerate(self.gate_kernels):
            gate_output = torch.einsum("ab,bc->ac", (feature_embedding, gate_kernel))
            if self.use_gate_bias:
                gate_output += self.gate_kernel_bias[index]
            gate_output = nn.Softmax(dim=-1)(gate_output)
            gate_outputs.append(gate_output)

        for gate_output in gate_outputs:
            expanded_gate_output = torch.unsqueeze(gate_output, 1)
            weighted_expert_output = expert_outputs * expanded_gate_output.expand_as(expert_outputs)
            final_outputs.append(torch.sum(weighted_expert_output, 2))
        return final_outputs
        click_prob, conversion_prob = torch.sigmoid(torch.squeeze(final_outputs[0])), torch.sigmoid(torch.squeeze(final_outputs[1]))
        return click_prob, conversion_prob
