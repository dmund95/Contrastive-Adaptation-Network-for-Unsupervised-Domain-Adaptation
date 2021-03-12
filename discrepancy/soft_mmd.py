from torch import nn
from utils.utils import to_cuda
import torch

class SMMD(object):
    def __init__(self, num_layers, kernel_num, kernel_mul, 
                 num_classes, intra_only=False, **kwargs):

        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.num_classes = num_classes
        self.intra_only = intra_only or (self.num_classes==1)
        self.num_layers = num_layers
    
    def split_classwise(self, dist, nums):
        num_classes = len(nums)
        start = end = 0
        dist_list = []
        for c in range(num_classes):
            start = end
            end = start + nums[c]
            dist_c = dist[start:end, start:end]
            dist_list += [dist_c]
        return dist_list

    def split_paired_dp_classwise(self, paired_dp, nums):
        num_classes = len(nums)
        start = end = 0
        ans_list = []
        for c in range(num_classes):
            start = end
            end = start + nums[c]
            paired_dp_c = paired_dp[start:end, start:end]
            ans_list += [paired_dp_c]
        return ans_list

    def gamma_estimation(self, dist):
        dist_sum = torch.sum(dist['ss']) + torch.sum(dist['tt']) + \
	    	2 * torch.sum(dist['st'])

        bs_S = dist['ss'].size(0)
        bs_T = dist['tt'].size(0)
        N = bs_S * bs_S + bs_T * bs_T + 2 * bs_S * bs_T - bs_S - bs_T
        gamma = dist_sum.item() / N 
        return gamma

    def patch_gamma_estimation(self, nums_S, nums_T, dist):
        assert(len(nums_S) == len(nums_T))
        num_classes = len(nums_S)

        patch = {}
        gammas = {}
        gammas['st'] = to_cuda(torch.zeros_like(dist['st'], requires_grad=False))
        gammas['ss'] = [] 
        gammas['tt'] = [] 
        for c in range(num_classes):
            gammas['ss'] += [to_cuda(torch.zeros([num_classes], requires_grad=False))]
            gammas['tt'] += [to_cuda(torch.zeros([num_classes], requires_grad=False))]

        source_start = source_end = 0
        for ns in range(num_classes):
            source_start = source_end
            source_end = source_start + nums_S[ns]
            patch['ss'] = dist['ss'][ns]

            target_start = target_end = 0
            for nt in range(num_classes):
                target_start = target_end 
                target_end = target_start + nums_T[nt] 
                patch['tt'] = dist['tt'][nt]

                patch['st'] = dist['st'].narrow(0, source_start, 
                       nums_S[ns]).narrow(1, target_start, nums_T[nt]) 

                gamma = self.gamma_estimation(patch)

                gammas['ss'][ns][nt] = gamma
                gammas['tt'][nt][ns] = gamma
                gammas['st'][source_start:source_end, \
                     target_start:target_end] = gamma

        return gammas

    def compute_kernel_dist(self, dist, gamma, kernel_num, kernel_mul, key):
        base_gamma = gamma / (kernel_mul ** (kernel_num // 2))
        gamma_list = [base_gamma * (kernel_mul**i) for i in range(kernel_num)]
        gamma_tensor = to_cuda(torch.stack(gamma_list, dim=0))

        eps = 1e-5
        gamma_mask = (gamma_tensor < eps).type(torch.cuda.FloatTensor)
        gamma_tensor = (1.0 - gamma_mask) * gamma_tensor + gamma_mask * eps 
        gamma_tensor = gamma_tensor.detach()

        for i in range(len(gamma_tensor.size()) - len(dist.size())):
            dist = dist.unsqueeze(0)

        dist = dist / gamma_tensor
        upper_mask = (dist > 1e5).type(torch.cuda.FloatTensor).detach()
        lower_mask = (dist < 1e-5).type(torch.cuda.FloatTensor).detach()
        normal_mask = 1.0 - upper_mask - lower_mask
        dist = normal_mask * dist + upper_mask * 1e5 + lower_mask * 1e-5
        kernel_val = torch.sum(torch.exp(-1.0 * dist), dim=0)
        return kernel_val

    def kernel_layer_aggregation(self, dist_layers, gamma_layers, key, category=None):
        num_layers = self.num_layers 
        kernel_dist = None
        for i in range(num_layers):

            dist = dist_layers[i][key] if category is None else \
                      dist_layers[i][key][category]

            gamma = gamma_layers[i][key] if category is None else \
                      gamma_layers[i][key][category]

            cur_kernel_num = self.kernel_num[i]
            cur_kernel_mul = self.kernel_mul[i]

            if kernel_dist is None:
                kernel_dist = self.compute_kernel_dist(dist, 
			gamma, cur_kernel_num, cur_kernel_mul, key) 

                continue

            kernel_dist += self.compute_kernel_dist(dist, gamma, 
                  cur_kernel_num, cur_kernel_mul, key) 

        return kernel_dist

    def patch_mean(self, nums_row, nums_col, dist, domain_probs_expand):
        assert(len(nums_row) == len(nums_col))
        num_classes = len(nums_row)
        num_domains = dist.size()[2]

        mean_tensor = to_cuda(torch.zeros([num_classes, num_classes, num_domains]))
        row_start = row_end = 0
        for row in range(num_classes):
            row_start = row_end
            row_end = row_start + nums_row[row]

            col_start = col_end = 0
            for col in range(num_classes):
                col_start = col_end
                col_end = col_start + nums_col[col]
                num = torch.sum(dist.narrow(0, row_start, nums_row[row]).narrow(1, col_start, nums_col[col]), [0,1])
                den = torch.sum(domain_probs_expand.narrow(0, row_start, nums_row[row]).narrow(1, col_start, nums_col[col]), [0,1])
                mean_tensor[row, col] = num/den
        return mean_tensor
        
    def compute_paired_dist(self, A, B):
        bs_A = A.size(0)
        bs_T = B.size(0)
        feat_len = A.size(1)

        A_expand = A.unsqueeze(1).expand(bs_A, bs_T, feat_len)
        B_expand = B.unsqueeze(0).expand(bs_A, bs_T, feat_len)
        dist = (((A_expand - B_expand))**2).sum(2)
        return dist
    
    def compute_soft_paired_dist(self, A, B, key, domain_probs, paired_domain_probs):
        bs_A = A.size(0)
        bs_T = B.size(0)
        feat_len = A.size(1)
        num_domains = domain_probs.size(1)

        A_expand = A.unsqueeze(1).expand(bs_A, bs_T, feat_len)
        B_expand = B.unsqueeze(0).expand(bs_A, bs_T, feat_len)
        dist = (((A_expand - B_expand))**2).sum(2)
        if key == 'ss':
            dist = dist.unsqueeze(2).expand(bs_A, bs_T, num_domains)
            return dist * paired_domain_probs # Ns x Ns x K
        elif key == 'st':
            dist = dist.unsqueeze(2).expand(bs_A, bs_T, num_domains)
            dp_expand = domain_probs.unsqueeze(1).expand(bs_A, bs_T, num_domains)
            return dist * dp_expand # Ns x Nt x K
        else:
            return dist # Nt x Nt


    def compute_paired_domain_prob(self, domain_probs):
        bs = domain_probs.size(0)
        num_domains = domain_probs.size(1)

        dp_1 = domain_probs.unsqueeze(1).expand(bs, bs, num_domains)
        dp_2 = domain_probs.unsqueeze(0).expand(bs, bs, num_domains)
        return dp_1 * dp_2

    def get_proper_labels(self, nums):
        num_classes = len(nums)
        ans = []
        for c in range(num_classes):
            ans.extend([c]*nums[c])
        return ans

    def forward(self, source, target, nums_S, nums_T, domain_probs):
        assert(len(nums_S) == len(nums_T)), \
             "The number of classes for source (%d) and target (%d) should be the same." \
             % (len(nums_S), len(nums_T))

        num_classes = len(nums_S)
        num_domains = domain_probs.size(2)
        # assert num_classes == domain_probs.size(1)

        proper_labels = self.get_proper_labels(nums_S)
        domain_probs_simple = domain_probs[torch.arange(domain_probs.size()[0]), proper_labels] # Ns x K
        paired_domain_probs = self.compute_paired_domain_prob(domain_probs_simple) # Ns x Ns x K

        paired_domain_probs_ss_classwise = self.split_paired_dp_classwise(paired_domain_probs, nums_S)

        # compute the dist 
        dist_layers = []
        gamma_layers = []

        for i in range(self.num_layers):

            cur_source = source[i]
            cur_target = target[i]

            dist = {}
            dist['ss'] = self.compute_paired_dist(cur_source, cur_source)
            dist['tt'] = self.compute_paired_dist(cur_target, cur_target)
            dist['st'] = self.compute_paired_dist(cur_source, cur_target)
            dist['ss'] = self.split_classwise(dist['ss'], nums_S)
            dist['tt'] = self.split_classwise(dist['tt'], nums_T)

            # soft_dist = {}
            # soft_dist['ss'] = self.compute_soft_paired_dist(cur_source, cur_source, 'ss', domain_probs, paired_domain_probs)
            # soft_dist['tt'] = self.compute_soft_paired_dist(cur_target, cur_target, 'tt', domain_probs, paired_domain_probs)
            # soft_dist['st'] = self.compute_soft_paired_dist(cur_source, cur_target, 'st', domain_probs, paired_domain_probs)
            # soft_dist['ss'] = self.split_classwise(soft_dist['ss'], nums_S)
            # soft_dist['tt'] = self.split_classwise(soft_dist['tt'], nums_T)

            dist_layers += [dist]

            gamma_layers += [self.patch_gamma_estimation(nums_S, nums_T, dist)]

        # compute the kernel dist
        for i in range(self.num_layers):
            for c in range(num_classes):
                gamma_layers[i]['ss'][c] = gamma_layers[i]['ss'][c].view(num_classes, 1, 1)
                gamma_layers[i]['tt'][c] = gamma_layers[i]['tt'][c].view(num_classes, 1, 1)

        kernel_dist_st = self.kernel_layer_aggregation(dist_layers, gamma_layers, 'st') # Ns x Nt
        assert kernel_dist_st.size()[0] == domain_probs_simple.size()[0]

        kernel_dist_st_expand = kernel_dist_st.unsqueeze(2).expand(kernel_dist_st.size()[0], kernel_dist_st.size()[1], num_domains) # Ns x Nt x K
        domain_probs_simple_expand = domain_probs_simple.unsqueeze(1).expand(kernel_dist_st.size()[0], kernel_dist_st.size()[1], num_domains) # Ns x Nt x K

        kernel_dist_st_soft = kernel_dist_st_expand * domain_probs_simple_expand
        kernel_dist_st_soft = self.patch_mean(nums_S, nums_T, kernel_dist_st_soft, domain_probs_simple_expand)

        kernel_dist_ss_soft = []
        kernel_dist_tt_soft = []
        for c in range(num_classes):
            kernel_dist_ss = self.kernel_layer_aggregation(dist_layers, gamma_layers, 'ss', c) # num_classes x N_c x N_c
            paired_dp_ss_c = paired_domain_probs_ss_classwise[c] # N_c x N_c x K

            kernel_dist_ss_expand = kernel_dist_ss.unsqueeze(2).expand(kernel_dist_ss.size()[0], kernel_dist_ss.size()[1], num_domains) # N_c x N_c x K

            temp_mult = kernel_dist_ss_expand * paired_dp_ss_c
            kernel_dist_ss_soft += [torch.sum(temp_mult.view(num_classes, num_domains, -1), dim=2) / torch.sum(paired_dp_ss_c.view(num_classes, num_domains, -1), dim=2)]

            temp_tt = torch.mean(self.kernel_layer_aggregation(dist_layers, gamma_layers, 'tt', c).view(num_classes, -1), dim=1)

            kernel_dist_tt_soft += [temp_tt.unsqueeze(1).expand(num_classes, num_domains)]

        kernel_dist_ss_soft = torch.stack(kernel_dist_ss_soft, dim=0)
        kernel_dist_tt_soft = torch.stack(kernel_dist_tt_soft, dim=0).transpose(1, 0)

        mmds = kernel_dist_ss_soft + kernel_dist_tt_soft - 2 * kernel_dist_st_soft

        intra = to_cuda(torch.zeros(1))
        inter = to_cuda(torch.zeros(1))

        for i in range(num_classes):
            for j in range(num_classes):
                if i == j:
                    intra += torch.mean(mmds[i,j])
                else:
                    inter += torch.mean(mmds[i,j])
        
        intra = intra / self.num_classes
        inter = inter / (self.num_classes * (self.num_classes - 1))

        cdd = intra if self.intra_only else intra - inter
        return {'cdd': cdd, 'intra': intra, 'inter': inter}

        # intra_mmds = torch.diag(mmds, 0)
        # intra = torch.sum(intra_mmds) / self.num_classes

        # inter = None
        # if not self.intra_only:
        #     inter_mask = to_cuda((torch.ones([num_classes, num_classes]) \
        #             - torch.eye(num_classes)).type(torch.ByteTensor))
        #     inter_mmds = torch.masked_select(mmds, inter_mask)
        #     inter = torch.sum(inter_mmds) / (self.num_classes * (self.num_classes - 1))

        # cdd = intra if inter is None else intra - inter
        # return {'cdd': cdd, 'intra': intra, 'inter': inter}
