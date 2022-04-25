import torch
import torch.nn as nn


class Expert(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(Expert, self).__init__()
        self.fc = nn.Linear(dim_input, dim_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class GatingNetwork(nn.Module):
    def __init__(self, dim_input, num_vectors):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(dim_input, num_vectors)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, selected_matrix, inputs):
        selector = self.softmax(self.fc(inputs))
        output = torch.einsum('bmd, bm -> bd', selected_matrix, selector)
        return output


class CGC(nn.Module):
    def __init__(self, dim_input, dim_experts_out, num_experts, num_tasks, is_top):
        super(CGC, self).__init__()
        self.dim_input = dim_input
        self.dim_experts_out = dim_experts_out
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.is_top = is_top

        experts = []
        gating_networks = []
        for i in range(self.num_tasks):
            experts_specific = nn.ModuleList(
                [Expert(self.dim_input, self.dim_experts_out) for i in range(self.num_experts[i])]
            )
            experts.append(experts_specific)
            gating_networks.append(GatingNetwork(dim_input, self.num_experts[-1] + self.num_experts[i]))
        experts_shared = nn.ModuleList(
            [Expert(self.dim_input, self.dim_experts_out) for i in range(self.num_experts[-1])]
        )
        experts.append(experts_shared)
        if not is_top:
            gating_networks.append(GatingNetwork(dim_input, sum(num_experts)))
        self.experts = nn.ModuleList(experts)
        self.gating_networks = nn.ModuleList(gating_networks)

    def forward(self, inputs):
        out_experts = []
        if len(inputs) != 1:
            for i in range(self.num_tasks):
                out_specific = [e(inputs[i]) for e in self.experts[i]]
                out_specific = torch.stack(out_specific, dim=1)
                out_experts.append(out_specific)
            out_shared = [e(inputs[-1]) for e in self.experts[-1]]
            out_shared = torch.stack(out_shared, dim=1)
            out_experts.append(out_shared)

            output = []
            for i in range(self.num_tasks):
                output.append(self.gating_networks[i](
                    selected_matrix=torch.cat((out_experts[i], out_experts[-1]), dim=1), inputs=inputs[i]
                ))
            if not self.is_top:
                output.append(self.gating_networks[-1](
                    selected_matrix=torch.cat(out_experts, dim=1), inputs=inputs[-1]
                ))
        else:
            for i in range(self.num_tasks):
                out_specific = [e(inputs[0]) for e in self.experts[i]]
                out_specific = torch.stack(out_specific, dim=1)
                out_experts.append(out_specific)
            out_shared = [e(inputs[0]) for e in self.experts[-1]]
            out_shared = torch.stack(out_shared, dim=1)
            out_experts.append(out_shared)

            output = []
            for i in range(self.num_tasks):
                output.append(self.gating_networks[i](
                    selected_matrix=torch.cat((out_experts[i], out_experts[-1]), dim=1), inputs=inputs[0]
                ))
            if not self.is_top:
                output.append(self.gating_networks[-1](
                    selected_matrix=torch.cat(out_experts, dim=1), inputs=inputs[0]
                ))

        return output


class Tower(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(dim_input, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class PLE(nn.Module):
    def __init__(self, num_layers, dim_x, dim_experts_out, num_experts, num_tasks, dim_tower, feature_vocabulary, embedding_size):
        super(PLE, self).__init__()
        self.num_layers = num_layers
        self.num_tasks = num_tasks

        self.feature_vocabulary = feature_vocabulary
        self.feature_names = sorted(list(feature_vocabulary.keys()))
        self.embedding_size = embedding_size
        self.embedding_dict = nn.ModuleDict()
        self.__init_weight()

        if num_layers == 1:
            layers = [CGC(dim_x, dim_experts_out[0], num_experts, num_tasks, True)]
        else:
            layers = [CGC(dim_x, dim_experts_out[0], num_experts, num_tasks, False)]
            for i in range(num_layers - 2):
                layers.append(CGC(dim_experts_out[i], dim_experts_out[i+1], num_experts, num_tasks, False))
            layers.append(CGC(dim_experts_out[-2], dim_experts_out[-1], num_experts, num_tasks, True))
        towers = []
        for i in range(num_tasks):
            towers.append(Tower(dim_tower[0], dim_tower[1], dim_tower[2]))
        self.layers = nn.ModuleList(layers)
        self.towers = nn.ModuleList(towers)

    def __init_weight(self, ):
        for name, size in self.feature_vocabulary.items():
            # emb = nn.Linear(size, self.embedding_size)
            emb = nn.Embedding(size, self.embedding_size)
            # nn.init.xavier_uniform(emb.weight)
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
            self.embedding_dict[name] = emb

    def forward(self, x):
        feature_embedding = []
        for name in self.feature_names:
            embed = self.embedding_dict[name](x[name])
            feature_embedding.append(embed)
        feature_embedding = torch.cat(feature_embedding, 1)  # [batch, 90]
        inputs = [feature_embedding]
        out = None
        for i in range(self.num_layers):
            out = self.layers[i](inputs)
            inputs = out
        result = []
        for i in range(self.num_tasks):
            result.append(self.towers[i](out[i]))
        click_pred, conversion_pred = torch.squeeze(result[0]), torch.squeeze(result[1])
        return click_pred, conversion_pred
