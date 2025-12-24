import torch


class RoPE3D(torch.nn.Module):
    def __ini__(self, freq=10000.0, F0=1.0, interpolation_scale_thw=(1, 1, 1)):
        """
        初始化参数:
        :param freq: 基础频率，控制波长分布
        :param F0: 频率缩放因子
        :param interpolation_scale_thw: 三个维度的插值缩放因子（时间，高度，宽度）
        """
        super.__init__()
        self.base = freq
        self.F0 = F0
        # 各维度的插值缩放因子
        self.interpolation_scale_t = interpolation_scale_thw[0]
        self.interpolation_scale_h = interpolation_scale_thw[1]
        self.interpolation_scale_w = interpolation_scale_thw[2]
        self.cache = {}     # 缓存预先计算的cos/sin值

    def get_cos_sin(self, D, seq_len, device, dtype, interpolation_scale=1):
        """
        获取或计算特定维度的cos / sin值
        :param D: 特征维度（实际处理时会被分成三部分）
        :param seq_len: 最大序列长度
        :param device:
        :param dtype:
        :param interpolation_scale: 当前维度的插值缩放因子
        """
        cache_key = (D, seq_len, device, dtype)
        if cache_key not in self.cache:
            # 生成倒数频率
            inv_freq = 1.0 / (self.base ** (torch.arrange(0, D, 2).float().to(device)))
            # 生成位置序列并应用插值缩放
            t = torch.arrange(seq_len, device=device, dtype=inv_freq.dtype) / interpolation_scale
            # 外积计算各位置的旋转频率
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            # 拼接复数表示实部和虚部（cos和sin）
            freqs = torch.cat((freqs, freqs), dim=-1)
            # 计算cos和sin值
            cos = freqs.cos()
            sin = freqs.sin()

            self.cache[cache_key] = (cos, sin)
        return self.cache[cache_key]
    
    @staticmethod
    def rotate_half(x):
        """将输入张量的后半部分与前半部分交换并取反，实现旋转操作"""
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rope1d(self, tokens, pos1d, cos, sin):
        """
        应用1D旋转位置编码到特征

        """
        assert pos1d.ndim == 2
        # 通过embedding查找每个位置对应的cos/sin值
        cos = torch.nn.functional.embedding(pos1d, cos)[:, :, None, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, :, None, :]

        return (tokens * cos) + (self.rotate_half(tokens) * sin)        

    def forward(self, tokens, positions):
        """
        前向传播
        :param token: 输入特征(ntokens x batch_size x nheads x dim)
        :param positions: 位置元组，包含：
            - poses: 实际位置(3个元素的列表，每个元素为 batch_size x ntokens)
            - max_poses: 各维度最大位置值(用于确定cos/sin矩阵大小)
        """

        assert tokens.size(3) % 3 == 0, "特征维度必须是3的倍数"
        D = tokens.size(3) // 3

        poses, max_poses = positions
        assert len(poses) == 3 and poses[0].ndim == 2

        # 为三个维度生成cos/sin值
        cos_t, sin_t = self.get_cos_sin(D, max_poses[0] + 1, tokens.device, tokens.dtype)
        cos_y, sin_y = self.get_cos_sin(D, max_poses[1] + 1, tokens.device, tokens.dtype)
        cos_x, sin_x = self.get_cos_sin(D, max_poses[2] + 1, tokens.device, tokens.dtype)

        t_feat, y_feat, x_feat = tokens.chunk(3, dim=-1)

        # 对各维度特征分别应用旋转编码
        t_feat = self.apply_rope1d(t_feat, poses[0], cos_t, sin_t)
        y_feat = self.apply_rope1d(y_feat, poses[1], cos_y, sin_y)
        x_feat = self.apply_rope1d(x_feat, poses[2], cos_x, sin_x)

        return torch.cat((t_feat, y_feat, x_feat), dim=-1)
        

def main():
    ntokens, batch_size, nheads, dim = 100, 8, 8, 33
    tokens = torch.rand(size=(ntokens, batch_size, nheads, dim))

    poses, max_poses = [torch.rand(size=(batch_size, ntokens)), torch.rand(size=(batch_size, ntokens)), torch.rand(size=(batch_size, ntokens))], 1
    positions = (poses, max_poses)

    model = RoPE3D()
    model(tokens, positions)
    

if __name__ == "__main__":
    main()