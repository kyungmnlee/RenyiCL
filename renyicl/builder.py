import torch
import torch.nn as nn
import numpy as np

def logmeanexp(inputs):
    return inputs.max() + (inputs - inputs.max()).exp().mean().log()

class RenyiCL(nn.Module):
    def __init__(self, backbone, alpha=1.0, dim=256, mlp_dim=4096, temp=0.5):
        super(RenyiCL, self).__init__()
        self.alpha = alpha
        self.temp = temp

        # build encoders
        self.source_encoder = backbone(num_classes=mlp_dim)
        self.target_encoder = backbone(num_classes=mlp_dim)
        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_s, param_t in zip(self.source_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_s.data)  # initialize
            param_t.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_s, param_t in zip(self.source_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = param_t.data * m + param_s.data * (1. - m)

    def extract_pos_neg(self, f_s, f_t):
        f_s = nn.functional.normalize(f_s, dim=1)
        f_t = nn.functional.normalize(f_t, dim=1)
        # gather all targets
        f_t = concat_all_gather(f_t)
        # Einstein sum is more intuitive        
        logits = torch.einsum('nc,mc->nm', [f_s, f_t]) / self.temp
        n = logits.shape[0]  # batch size per GPU
        indx = torch.distributed.get_rank()
        logits_chunk = logits.chunk(torch.distributed.get_world_size(), dim=1)
        logits_current = logits_chunk[indx]
        pos = logits_current.diag().view(-1, 1)
        neg = logits_current.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)
        for u in range(torch.distributed.get_world_size()):
            if u != indx:
                neg = torch.cat((neg, logits_chunk[u]), dim=1)
        return pos, neg

    def contrastive_loss(self, pos, neg, gamma):
        pos_sg = pos.clone().detach()
        pos_sg_all = concat_all_gather(pos_sg)
        neg_sg = neg.clone().detach()
        neg_sg_all = concat_all_gather(neg_sg)

        if gamma == 1:
            loss_1 = -pos.mean()
        else:
            e_pos_1 = ((gamma-1)*pos_sg).exp()
            e_pos_all_1 = ((gamma-1) * pos_sg_all).exp()
            denom_1 = e_pos_all_1.mean()
            loss_1 = -torch.mean(pos * e_pos_1) / denom_1

        e_pos_2 = (gamma * pos_sg).exp()
        e_neg_2 = (gamma * neg_sg).exp()
        e_pos_all_2 = (gamma * pos_sg_all).exp()
        e_neg_all_2 = (gamma * neg_sg_all).exp()
        denom_2 = self.alpha * e_pos_all_2.mean() + (1-self.alpha)*e_neg_all_2.mean()
        num_1 = torch.mean(pos * e_pos_2.detach())
        num_2 = torch.mean(neg * e_neg_2.detach())
        loss_2 = (self.alpha * num_1 + (1 - self.alpha) * num_2) / denom_2
        loss = loss_1 + loss_2
        return loss

    @torch.no_grad()
    def mutual_information(self, pos, neg):
        pos_sg = pos.clone().detach()
        pos_sg_all = concat_all_gather(pos_sg)
        neg_sg = neg.clone().detach()
        neg_sg_all = concat_all_gather(neg_sg)

        with torch.no_grad():
            Z = torch.log(self.alpha * pos_sg_all.exp().mean(dim=1) + (1 - self.alpha) * neg_sg_all.exp().mean(dim=1))
            r_alpha = (pos_sg_all.mean(dim=1) - Z).exp()
            r_true = (1 - self.alpha) * r_alpha / (1 - self.alpha * r_alpha)
            log_ratio = torch.log(torch.clamp(r_true, min=1e-10))
            mi = log_ratio.mean()
        return mi

    def forward(self, images, m, gamma):
        # compute features
        bs = images[0].size(0)        
        f_s, f_t = [], []
        for img in images:
            f_s.append(self.predictor(self.source_encoder(img)))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            f_t.append(self.target_encoder(images[0]))
            f_t.append(self.target_encoder(images[1]))

        loss = 0.0
        mi = 0.0
        for idx_t in range(2):
            pos, neg = [], []
            for idx_s in range(len(images)):
                if idx_s == idx_t:
                    continue
                pos_, neg_ = self.extract_pos_neg(f_s[idx_s], f_t[idx_t])
                pos.append(pos_)
                neg.append(neg_)
            pos = torch.cat(pos, dim=1)
            neg = torch.cat(neg, dim=1)
            loss += self.contrastive_loss(pos, neg, gamma)
            mi += self.mutual_information(pos, neg)

        return loss, mi/2

class RenyiCL_ResNet(RenyiCL):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.source_encoder.fc.weight.shape[1]
        del self.source_encoder.fc, self.target_encoder.fc # remove original fc layer

        # projectors
        self.source_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.target_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)

class MoCov3(nn.Module):
    def __init__(self, backbone, dim=256, mlp_dim=4096, temp=0.5):
        super(MoCov3, self).__init__()
        self.temp = temp

        # build encoders
        self.source_encoder = backbone(num_classes=mlp_dim)
        self.target_encoder = backbone(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_s, param_t in zip(self.source_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_s.data)  # initialize
            param_t.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_s, param_t in zip(self.source_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = param_t.data * m + param_s.data * (1. - m)

    def extract_pos_neg(self, f_s, f_t):
        f_s = nn.functional.normalize(f_s, dim=1)
        f_t = nn.functional.normalize(f_t, dim=1)
        # gather all targets
        f_t = concat_all_gather(f_t)
        # Einstein sum is more intuitive        
        logits = torch.einsum('nc,mc->nm', [f_s, f_t]) / self.temp
        n = logits.shape[0]  # batch size per GPU
        indx = torch.distributed.get_rank()
        logits_chunk = logits.chunk(torch.distributed.get_world_size(), dim=1)
        logits_current = logits_chunk[indx]
        pos = logits_current.diag().view(-1, 1)
        neg = logits_current.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)
        for u in range(torch.distributed.get_world_size()):
            if u != indx:
                neg = torch.cat((neg, logits_chunk[u]), dim=1)
        return pos, neg

    def contrastive_loss(self, q, k):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.temp
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.temp)

    @torch.no_grad()
    def mutual_information(self, f_s, f_t):
        pos, neg = self.extract_pos_neg(f_s, f_t)
        alpha = 1 / f_s.size(0)

        pos_sg = pos.clone().detach()
        pos_sg_all = concat_all_gather(pos_sg)
        neg_sg = neg.clone().detach()
        neg_sg_all = concat_all_gather(neg_sg)

        with torch.no_grad():
            Z = torch.log(alpha * pos_sg_all.exp().mean(dim=1) + (1 - alpha) * neg_sg_all.exp().mean(dim=1))
            r_alpha = (pos_sg_all.mean(dim=1) - Z).exp()
            r_true = (1 - alpha) * r_alpha / (1 - alpha * r_alpha)
            log_ratio = torch.log(torch.clamp(r_true, min=1e-10))
            mi = log_ratio.mean()
        return mi

    def forward(self, images, m):
        # compute features
        bs = images[0].size(0)        
        f_s, f_t = [], []
        for img in images:
            f_s.append(self.predictor(self.source_encoder(img)))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            f_t.append(self.target_encoder(images[0]))
            f_t.append(self.target_encoder(images[1]))

        loss = 0.0
        mi = 0.0
        for idx_t in range(2):
            for idx_s in range(len(images)):
                if idx_s == idx_t:
                    continue
                loss += self.contrastive_loss(f_s[idx_s], f_t[idx_t])
                mi   += self.mutual_information(f_s[idx_s], f_t[idx_t])
        loss = loss / (len(images) - 1)
        mi = mi / (len(images) - 1)

        return loss, mi/2

class MoCov3_ResNet(MoCov3):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.source_encoder.fc.weight.shape[1]
        del self.source_encoder.fc, self.target_encoder.fc # remove original fc layer

        # projectors
        self.source_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.target_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

