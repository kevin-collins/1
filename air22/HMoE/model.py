import torch
import torch.nn as nn
from MMoE import MMoE


class Attention(nn.Module):
  def __init__(self, dim=5):
    super(Attention, self).__init__()
    self.dim = dim
    self.q_layer = nn.Linear(dim, dim, bias=False)
    self.k_layer = nn.Linear(dim, dim, bias=False)
    self.v_layer = nn.Linear(dim, dim, bias=False)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, inputs):
    Q = self.q_layer(inputs)
    K = self.k_layer(inputs)
    V = self.v_layer(inputs)
    a = torch.sum(torch.mul(Q, K), -1) / torch.sqrt(torch.tensor(self.dim, dtype=torch.float64))
    a = self.softmax(a)
    outputs = torch.sum(torch.mul(torch.unsqueeze(a, -1), V), dim=1)
    return outputs
"""
class MMoE(torch.nn.Module):
    def __init__(self, args):
        super(MMoE, self).__init__()
        self.scenario_nums = args.scenario_nums
        self.input_size = args.input_size
        self.expert_size = args.expert_size
        self.expert_nums = args.expert_nums
        self.gate_size = args.gate_size
        self.tower_size = args.tower_size
        self.tower_size2 = args.tower_size2
        self.device = args.device
        self.experts = [
            nn.Sequential(
                nn.Linear(self.input_size, self.expert_size),
                nn.ReLU(),
            ).to(args.device) for _ in range(self.expert_nums)
        ]

        self.gates = [nn.Sequential(
                nn.Linear(self.input_size, self.gate_size),
                nn.Linear(self.gate_size, self.gate_size),
                nn.Linear(self.gate_size, self.expert_nums),
                nn.Softmax(dim=1)
            ).to(args.device) for _ in range(self.scenario_nums)
        ]

        self.towers = [
            nn.Sequential(
                nn.Linear(self.expert_size, self.tower_size),
                nn.ReLU(),
                nn.Linear(self.tower_size, self.tower_size2),
                nn.ReLU(),
                nn.Linear(self.tower_size2, 1),
            ).to(args.device) for _ in range(self.scenario_nums)
        ]

    def forward(self, x, i):
        # 根据场景选择对应的门  [batch,1,experts]
        gate = torch.unsqueeze(self.gates[i](x), dim=1)
        # gate = torch.unsqueeze(self.gates(x), dim=1)
        # 获取专家网络的输出  [batch,expert,expert_size]
        experts_out = torch.stack([self.experts[j](x) for j in range(self.expert_nums)], dim=1)

        # 加权和   [batch,expert_size]
        experts_out_gated = torch.squeeze(torch.bmm(gate, experts_out))

        # 塔网络计算得分 sigmoid  [batch]
        s = torch.squeeze(self.towers[i](experts_out_gated))
        return s

"""


class HMoE(torch.nn.Module):
    def __init__(self, args, feature_vocabulary, embedding_size):
        super(HMoE, self).__init__()
        self.feature_vocabulary = feature_vocabulary
        self.feature_names = sorted(list(feature_vocabulary.keys()))
        self.embedding_size = embedding_size
        self.embedding_dict = nn.ModuleDict()
        self.device = args.device
        self.__init_weight()

        self.scenario_nums = args.scenario_nums
        self.input_size = args.input_size
        self.expert_size = args.expert_size
        self.expert_nums = args.expert_nums
        self.gate_size = args.gate_size
        self.tower_size = args.tower_size

        self.factor_nums = args.factor_nums
        self.factor_size = args.factor_size

        self.click_model = MMoE(args.input_size, 1, args.expert_nums, args.scenario_nums, args.device, feature_vocabulary)
        self.purchase_model = MMoE(args.input_size, 1, args.expert_nums, args.scenario_nums, args.device, feature_vocabulary)

        self.gate = nn.Sequential(
                nn.Linear(self.input_size, self.gate_size),
                nn.Linear(self.gate_size, self.gate_size),
                nn.Linear(self.gate_size, self.scenario_nums),
                nn.Softmax(dim=1)
        )
        """
        self.attention = Attention(self.input_size)
        self.e2f = [
            nn.Sequential(
                nn.Linear(self.input_size, self.factor_size),
            ).to(self.device) for _ in range(self.factor_nums)
        ]
        self.gate_choose = nn.Sequential(
            nn.Linear(self.input_size, self.gate_size),
            nn.Linear(self.gate_size, self.factor_nums),
            nn.Sigmoid(),
        )

        
        self.f2e = nn.Sequential(
            nn.Linear(self.factor_size*self.factor_nums, self.factor_size),
            nn.ReLU(),
            nn.Linear(self.factor_size, self.factor_size),
            nn.ReLU(),
            )
        """
        self.sigmoid = nn.Sigmoid()
    def __init_weight(self, ):
        for name, size in self.feature_vocabulary.items():
            emb = nn.Embedding(size, self.embedding_size)
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
            self.embedding_dict[name] = emb

    def forward(self, x):
        # 生成特征

        feature_embedding_ori = []
        for name in self.feature_names:
            embed = self.embedding_dict[name](x[name])
            feature_embedding_ori.append(embed)
        feature_embedding = torch.cat(feature_embedding_ori, 1)  # [batch, 90]


        # [batch, 3, 90]
        """
        factors = torch.stack([self.e2f[i](feature_embedding_ori) for i in range(self.factor_nums)], dim=1)
        gate_choose = torch.unsqueeze(self.gate_choose(feature_embedding_ori), dim=2)  # [batch, 3, 1]
        factors_gated = factors * gate_choose  # [batch, 3, 90] boardcost
        # feature_embedding = self.f2e(factors_gated.view(-1, self.factor_nums*self.factor_size))
        feature_embedding = self.attention(factors_gated)  # [batch, 90]
        """
        nums = feature_embedding.size()[0]

        _ctrs = self.click_model(feature_embedding)
        _cvrs = self.purchase_model(feature_embedding)

        """
        _ctrs = [None for _ in range(self.scenario_nums)]
        _cvrs = [None for _ in range(self.scenario_nums)]
        for i in range(self.scenario_nums):  # 样本每个场景下的分数
            _ctrs[i] = self.click_model(feature_embedding, i)
            _cvrs[i] = self.purchase_model(feature_embedding, i)

        
        _ctrs = [torch.ones_like(__ctrs[0]) for _ in range(self.scenario_nums)]
        _cvrs = [torch.ones_like(__cvrs[0]) for _ in range(self.scenario_nums)]

        # 梯度截断
        for i in range(nums):
            scenario = int(x['301'][i])-1  # 当前场景号 1，2，3
            for j in range(self.scenario_nums):
                if scenario != j:
                    _ctrs[j][i] = __ctrs[j][i].detach()
                    _cvrs[j][i] = __cvrs[j][i].detach()
                else:
                    _ctrs[j][i] = __ctrs[j][i]
                    _cvrs[j][i] = __cvrs[j][i]
        """
        # 组成tensor
        ctrs = torch.stack(_ctrs, dim=1)  # [batch,s,1]
        cvrs = torch.stack(_cvrs, dim=1)  # , dim=2)

        # 得到标签域W
        gate = torch.unsqueeze(self.gate(feature_embedding), dim=1)  # [batch,1,s]

        # 加权和
        pred_ctr = self.sigmoid(torch.bmm(gate, ctrs))  # [batch,1]
        pred_cvr = self.sigmoid(torch.bmm(gate, cvrs))

        return torch.squeeze(pred_ctr), torch.squeeze(pred_cvr), None, None  # [batch]

