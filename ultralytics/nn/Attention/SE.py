import torch.nn as nn


class SE_block(nn.Module):
    def __init__(self, channel, scaling=16, use_inplace=True):
        """
        SE注意力模块
        :param channel: 输入特征图的通道数
        :param scaling: 中间层的缩放比例，默认为16
        :param use_inplace: 是否使用inplace操作，默认为True
        """
        super(SE_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // scaling, bias=False),  # 压缩通道
            nn.ReLU(inplace=use_inplace),  # 激活函数
            nn.Linear(channel // scaling, channel, bias=False),  # 恢复通道
            nn.Sigmoid()  # 生成注意力权重
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: 全局平均池化并展平
        y = self.avg_pool(x).flatten(1)
        # Excitation: 通过全连接层生成注意力权重
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: 对输入特征图进行重标定
        return x * y