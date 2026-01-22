import torch.nn as nn
import torch


class CCC(nn.Module):
    def __init__(self):
        super(CCC, self).__init__()
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, y_true, y_pred):
        loss = 1 - self.cos(y_pred - y_pred.mean(dim=0, keepdim=True), y_true - y_true.mean(dim=0, keepdim=True))
        return loss  # 返回 1 减去 CCC 的值作为损失函数



class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
     



    def get_x_V_P(self, batch_x, batch_y, batch_size, c_norms, device):
        """
        引入谓词,构造V和P矩阵
        问题：p矩阵和V矩阵是否需要归一化？
        """
        V = torch.eye(batch_size, device=device)  # 单位矩阵 (batch_size x batch_size)

        phi = batch_x[:,:,1].mean(dim=1)
        phi = (phi / torch.linalg.norm(phi)).unsqueeze(0)
        P = torch.mm( phi.T,phi)    

        return V, P


    def forward(self, batch_x, outputs, targets, tau_hat=None, tau=None, c_norms=None):
        batch_size = outputs.shape[0]
        device = outputs.device

        V, P = self.get_x_V_P(batch_x, targets, batch_size, c_norms, device) # phi=other
        # P = P + P_ones
        error = outputs - targets
        if tau_hat is None or tau is None:
            weighted_error = error
            return {
                # 'total':  torch.nn.functional.mse_loss(outputs, targets),
                'total':  torch.mean(weighted_error ** 2),
                'mse': torch.mean(weighted_error ** 2),
                'V_loss': torch.mean(torch.matmul(V, error)** 2),
                'P_loss': torch.mean(torch.matmul(P, error)** 2),
            }
        else:
            weight_matrix = tau_hat * V + tau * P # V保证自我, P batch内特征相似度 相似度越高对应P里面位置的数越大，如果这时候误差很大就会放大误差做惩罚
            weighted_error = torch.matmul(weight_matrix, error)
            return {
                'total': torch.mean(weighted_error ** 2),
                'mse': torch.mean(weighted_error ** 2),
                'V_loss': torch.mean(torch.matmul(tau_hat * V, error)** 2),
                'P_loss': torch.mean(torch.matmul(tau * P, error)** 2), # 特征相似度的误差
                }