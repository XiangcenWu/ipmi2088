import torch
import torch.nn as nn


device = "cuda" if torch.cuda.is_available() else 'cpu'


class SelectionNet(nn.Module):

    def __init__(self, DS_dim_list, num_resblock, transformer_input_dim, num_head, dropout=0, num_transformer=1):
        super().__init__()
        self.down_sample = DownSample(
            DS_dim_list,
            num_resblock
        )
        self.pool_to_vector = nn.Sequential(
            nn.Conv2d(transformer_input_dim, transformer_input_dim, 5, 5),
            nn.AvgPool2d(2)
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(transformer_input_dim, num_head, dropout=dropout, batch_first=True), 
            num_transformer
        )
        self.output = nn.Sequential(
            nn.Linear(transformer_input_dim, transformer_input_dim // 2),
            nn.ReLU(),
            nn.Linear(transformer_input_dim // 2, transformer_input_dim // 4),
            nn.ReLU(),
            nn.Linear(transformer_input_dim // 4, transformer_input_dim // 8),
            nn.Linear(transformer_input_dim // 8, 1),
        )

    def forward(self, x):
        
        feature = self.down_sample(x)
        # print(feature.shape)
        feature = self.pool_to_vector(feature).flatten(1)
        # print(feature.shape)


        feature = feature.unsqueeze(0)
        # print(feature.shape)
        feature = self.transformer(feature)
        # print(feature.shape)
        return self.output(feature).squeeze(0).permute(0, 1)




class DownSample(nn.Module):

    def __init__(self, feature_list=[1, 32, 128, 256, 512, 1024], num_resblock=2):
        super().__init__()
        self.num_resblock = num_resblock
        self.feature_list = feature_list
        self.down_0 = self._make_down(0)
        self.down_1 = self._make_down(1)
        self.down_2 = self._make_down(2)
        self.down_3 = self._make_down(3)
        self.down_4 = self._make_down(4)

        


    def forward(self, x):
        x = self.down_0(x)
        x = self.down_1(x)
        x = self.down_2(x)
        x = self.down_3(x)
        x = self.down_4(x)

        return x


    def _make_down(self, i_th_block):
        return nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(self.feature_list[i_th_block], self.feature_list[i_th_block+1], 1),
            ResBlock(self.feature_list[i_th_block+1], self.num_resblock),

        )


class ResBlock(nn.Module):

    def __init__(self, num_in, num_blocks):
        super().__init__()
        self.relu = nn.ReLU()
        self.main_block = nn.ModuleList([self._creat_block(num_in) for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.main_block:
            x = self.relu(x + block(x))
        return x


    def _creat_block(self, num_in):
        # up is the resnet version down is more param version
        return nn.Sequential(
            nn.Conv2d(num_in, num_in, 1),
            nn.Conv2d(num_in, num_in, 3, 1, 1),
            nn.Conv2d(num_in, num_in, 1)
        )
