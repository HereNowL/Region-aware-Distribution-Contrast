import torch.nn as nn
import torch as torch
import torch.nn.functional as F
# from  model.mapfns import conv_task
import model.config_task as config_task
import numpy as np
import cProfile
class conv_task(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1, kernel_size=3, padding=1, num_tasks=2):
        super(conv_task, self).__init__()
        self.num_tasks = num_tasks
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, padding=1)
        self.gamma = nn.Parameter(torch.ones(planes, num_tasks*(num_tasks-1)))
        self.beta = nn.Parameter(torch.zeros(planes, num_tasks*(num_tasks-1)))
        self.bn = nn.BatchNorm2d(num_features=planes)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):

        # first, get the taskpair information: compute A
        A_taskpair = config_task.A_taskpair

        x = self.conv(x)

        # generate taskpair-specific FiLM parameters
        gamma = torch.mm(A_taskpair, self.gamma.t())
        beta = torch.mm(A_taskpair, self.beta.t())
        gamma = gamma.view(1, x.size(1), 1, 1)
        beta = beta.view(1, x.size(1), 1, 1)

        x = self.bn(x)

        # taskpair-specific transformation
        x = x * gamma + beta
        x = self.relu(x)


        return x

def trace(A):
    return A.diagonal(dim1=-2, dim2=-1).sum(-1)

def sqrt_newton_schulz_autograd(A, numIters, dtype, eps=1e-8):
    A = A.unsqueeze(0)
    batchSize = A.data.shape[0]
    dim = A.data.shape[1]
    normA = torch.sqrt(A.mul(A).sum(dim=1).sum(dim=1)+eps)
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
    I = Variable(torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype), requires_grad=False).cuda()
    Z = Variable(torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype), requires_grad=False).cuda()

    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    sA = Y * torch.sqrt(normA+eps).view(batchSize, 1, 1).expand_as(A)
    sA = sA.squeeze(0)
    return sA

def cov_product_sqrt(sigma_p, sigma_q, eps=1e-8):
    pmmq = torch.matmul(sigma_p, sigma_q)
    trace = torch.trace(pmmq)
    cps = torch.pow((torch.abs(trace) + eps), 1/2)

    return cps

class Region_Contra_Loss(nn.Module):
    def __init__(self, tasks=['semantic', 'depth', 'normal'], last_inp_channels=512) -> None:
        super(Region_Contra_Loss, self).__init__()
        # self.conv = nn.
        self.eps = 1e-6
        self.contra_temp = 0.5
        self.simi_temp = 0.4
        self.feature_bank = []
        self.capacity = 4
        self.count = 0
        self.edge_feature_bank = []
        self.nonedge_feature_bank = []
        
        self.feature_bank_u = []
        self.feature_bank_s = []
        self.bankR_cnt = 1000

    def forward(self, map_s, map_t, mask=None, index=None):
        
        
        region_list_x, ux_list, sx_list, region_list_y, uy_list, sy_list = self.get_region_numpy(map_s, map_t, mask)
        
        self.feature_bank_u += ux_list + uy_list
        self.feature_bank_s += sx_list + sy_list
        #添加feature bank 代码
        self.feature_bank_u = self.feature_bank_u[-self.bankR_cnt :]
        self.feature_bank_s = self.feature_bank_s[-self.bankR_cnt :]
        # print(len(self.feature_bank_u))
        
        ###########

        loss = 0
        numR = len(region_list_x)
        for i in range(numR):
            loss += self.comp_constra_loss_gaussian(region_list_x[i], region_list_y[i], [ux_list[i], sx_list[i]], [uy_list[i], sy_list[i]],[self.feature_bank_u, self.feature_bank_s])
            
        loss = loss / numR
    
        # print('loss: {}'.format(loss))
        return loss        


    #x: (1, dim, rows, cols)
    def get_region_tensor(self, x, y, sam):

        samv = sam[0]
        
        region_list_x, region_list_y = [], []
        
        ux_list, uy_list, sx_list, sy_list = [], [], [], []

        for i in range(256):
            
            index = (samv == i).nonzero(as_tuple=True)
            
            if len(index[0]) < 10:
                continue
            
            region_x = x[0, :, index[0], index[1]]
            region_y = y[0, :, index[0], index[1]]
            
            u_x = torch.mean(region_x, dim = 1)
            sigma_x = torch.cov(region_x)
            u_y = torch.mean(region_y, dim = 1)
            sigma_y = torch.cov(region_y)
            
            region_list_x.append(region_x)
            region_list_y.append(region_y)
            ux_list.append(u_x)
            sx_list.append(sigma_x)
            uy_list.append(u_y)
            sy_list.append(sigma_y)            
        

        return region_list_x, ux_list, sx_list, region_list_y, uy_list, sy_list

    def get_region_numpy(self, x, y, sam):

        samv = sam[0]
        
        region_list_x, region_list_y = [], []
        
        ux_list, uy_list, sx_list, sy_list = [], [], [], []

        for i in range(256):
            
            index = (samv == i).nonzero(as_tuple=True)
            
            if len(index[0]) < 10:
                continue
            
            region_x = x[0, :, index[0], index[1]]
            region_y = y[0, :, index[0], index[1]]
            
            u_x = np.mean(region_x.detach().cpu().numpy(), axis = 1)
            sigma_x = np.cov(region_x.detach().cpu().numpy())
            u_y = np.mean(region_y.detach().cpu().numpy(), axis = 1)
            sigma_y = np.cov(region_y.detach().cpu().numpy())
            
            region_list_x.append(region_x)
            region_list_y.append(region_y)
            ux_list.append(u_x)
            sx_list.append(sigma_x)
            uy_list.append(u_y)
            sy_list.append(sigma_y)            
        

        return region_list_x, ux_list, sx_list, region_list_y, uy_list, sy_list

#     def comp_constra_loss_gaussian(self, anchor, pos_pair, an_vs, n_vs, neg_vs, temp = 1):
#         pos = torch.div(-self.w_dist(an_vs[0], an_vs[1], n_vs[0], n_vs[1]), 5.4)  
#         maxp = torch.max(pos, dim = 0, keepdim = True)[0]   
#         negNum = len(neg_vs)
        
#         neg = 0
#         for i in range(negNum):
#             neg += torch.exp(-self.w_dist(an_vs[0], an_vs[1], neg_vs[0][i], neg_vs[1][i]) / 5.4)  
            
#         # loss = pos / (neg + self.eps)
#         # loss = torch.log(loss + self.eps)
#         # loss = loss.mean()
#         exp_pos = torch.exp(pos - maxp).mean()
        
# #        exp_neg = torch.exp(-neg / negNum)
#         exp_neg = neg / negNum
        
#         loss = exp_pos / (exp_neg + self.eps)
        
#         loss = -torch.log(loss + self.eps)
#         output = loss.detach().item()
#         del loss
#         return output

    def comp_constra_loss_gaussian(self, anchor, pos_pair, an_vs, n_vs, neg_vs, temp = 1):
        pos = - self.calculate_frechet_distance(an_vs[0], an_vs[1], n_vs[0], n_vs[1]) / 5.4 
        maxp = np.max(pos, axis= 0, keepdims=True)
        exp_pos = np.exp(pos - maxp).mean()
        negNum = len(neg_vs)
        
        neg = 0
        for i in range(negNum):
            neg += np.exp(-self.calculate_frechet_distance(an_vs[0], an_vs[1], neg_vs[0][i], neg_vs[1][i]) / 5.4)  

        exp_neg = neg / negNum
        
        loss = exp_pos / (exp_neg + self.eps)
        
        loss = -np.log(loss + self.eps)
        # output = loss.detach().item()
        # del loss
        return loss

    
    def w_dist(self, mu_p, sigma_p, mu_q, sigma_q):
        import math
        # 计算均值差的范数的平方
        mean_diff_sq = torch.norm(mu_p - mu_q) ** 2
        # 计算协方差矩阵的迹
        trace_sp = torch.trace(sigma_p)
        trace_sq = torch.trace(sigma_q)
        
        cps = cov_product_sqrt(sigma_p=sigma_p, sigma_q=sigma_q)

        wasserstein_sq = mean_diff_sq + trace_sp + trace_sq - 2 * cps

        return wasserstein_sq

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        from scipy import linalg
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        offset = np.eye(sigma1.shape[0]) * eps
        covmean, _ = linalg.sqrtm(np.dot(sigma1+offset, sigma2+offset), disp=False)

        covmean = np.abs(covmean)

        tr_covmean = np.trace(covmean)

        term1 = diff.dot(diff) + np.trace(sigma1)+ np.trace(sigma2) 
        term2 = - 2 * tr_covmean
        dist = term1 + term2

        # print('term1: {}'.format(term1))
        # print('term2: {}'.format(term2))

        return dist  
    
class RegionConsistency(nn.Module):
    def __init__(self, tasks=['semantic', 'depth', 'normal'], last_inp_channels=512) -> None:
        super(RegionConsistency, self).__init__()
        self.num_tasks = len(tasks)
        self.last_layer = nn.Sequential(self.conv_layer([last_inp_channels, last_inp_channels]),
                                        self.conv_layer([last_inp_channels, 32]))
        
        self.comp_contrastive_loss = Region_Contra_Loss()

        
    def forward(self, map_s, map_t, img_size, mask=None, index=None):
        # map_s = self.last_layer(map_s)
        # map_t = self.last_layer(map_t)
        
        # (1, 128, 144, 192)
        map_s = F.interpolate(map_s, img_size, mode='bilinear')
        map_t = F.interpolate(map_t, img_size, mode='bilinear')
        mask = F.interpolate(mask.unsqueeze(0), img_size, mode='nearest').squeeze(0)

        loss = self.comp_contrastive_loss(map_s, map_t, mask, index)

        return loss
    
    def conv_layer(self, channel, num_tasks=None):
        return conv_task(in_planes=channel[0], planes=channel[1], kernel_size=1, stride=1, padding=1, num_tasks=self.num_tasks)



