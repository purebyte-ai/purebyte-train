"""Causal depthwise conv1d with carried state (ssm_v2 step 4, spec/FORMAT.md §9.1)."""
import torch.nn as nn


class CausalConv1d(nn.Module):
    """Depthwise over the xBC channels. The state is the last K-1 input rows. Exported as `conv1d.weight/bias`."""

    def __init__(self, channels: int, width: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, width, groups=channels)
        self.channels, self.width = channels, width

    def forward(self, x, state=None):
        """x (B,T,C); state (B,C,K-1) or None -> (y (B,T,C), state_out (B,C,K-1))."""
        import torch
        b, t, c = x.shape
        xt = x.transpose(1, 2)                                   # (B,C,T)
        if state is None:
            state = xt.new_zeros(b, c, self.width - 1)
        xp = torch.cat([state, xt], dim=-1)                      # (B,C,K-1+T)
        y = self.conv(xp)                                        # (B,C,T)
        return y.transpose(1, 2), xp[..., -(self.width - 1):]
