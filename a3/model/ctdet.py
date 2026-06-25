from modules.centernet.ctdet import *
from torchvision.models import resnet18, ResNet18_Weights


class Model(nn.Module):
    def __init__(self, num_classes, topk):
        super().__init__()
        self.num_classes = num_classes
        self.topk = topk
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            *list(self.backbone.children())[:-2]
        )  # Not include avg_pool and last linear layer
        self.smooth = nn.Sequential(
            SPP(),
            ConvLayer(
                512 * 4, 256, kernel_size=1, padding=0
            ),  # 512 * 4 is the channel dimension of the last conv layer output, c5, from resnet
            ConvLayer(256, 512, kernel_size=3, padding=1),
        )
        self.deconv5 = DeConvLayer(512, 256, kernel_size=4, stride=2)
        self.deconv4 = DeConvLayer(256, 256, kernel_size=4, stride=2)
        self.deconv3 = DeConvLayer(256, 256, kernel_size=4, stride=2)

        self.cls_pred = nn.Sequential(
            ConvLayer(256, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, self.num_classes, kernel_size=1),
        )

        self.txty_pred = nn.Sequential(
            ConvLayer(256, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 2, kernel_size=1),
        )

        self.twth_pred = nn.Sequential(
            ConvLayer(256, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 2, kernel_size=1),
        )

    def forward(self, x):
        c5 = self.backbone(x)
        N = c5.shape[0]
        p5 = self.smooth(c5)
        p4 = self.deconv5(p5)
        p3 = self.deconv4(p4)
        p2 = self.deconv3(p3)

        # If input image x size is (3, 448, 448) p2 fm shape is (256, 112, 112).
        # Therefore, the downsampling factor is 4
        # print('p2.shape', p2.shape)

        cls_pred = self.cls_pred(p2)
        txty_pred = self.txty_pred(p2)
        twth_pred = self.twth_pred(p2)

        # (N, C, H, W) to (N, H*W, C)
        cls_pred = (
            cls_pred.permute(0, 2, 3, 1).contiguous().view(N, -1, self.num_classes)
        )  # heatmap
        # (N, 2, H, W) to (N, H*W, 2)
        txty_pred = (
            txty_pred.permute(0, 2, 3, 1).contiguous().view(N, -1, 2)
        )  #  offsets: dx, dy
        # (N, 2, H, W) to (N, H*W, 2)
        twth_pred = twth_pred.permute(0, 2, 3, 1).contiguous().view(N, -1, 2)  # wh

        return cls_pred, txty_pred, twth_pred

    def get_topk(self, scores):
        N, C, H, W = scores.shape
        topk_scores, topk_inds = torch.topk(scores.view(N, C, -1), self.topk)
        topk_inds = topk_inds % (H * W)
        topk_score, topk_ind = torch.topk(topk_scores.view(N, -1), self.topk)
        topk_inds = self.gather_feat(topk_inds.view(N, -1, 1), topk_ind).view(
            N, self.topk
        )
        topk_clses = torch.floor_divide(topk_ind, self.topk).int()
        return topk_score, topk_inds, topk_clses

    def gather_feat(self, feat, ind):
        dim = feat.shape[2]
        ind = ind.unsqueeze(2).expand(ind.shape[0], ind.shape[1], dim)
        return feat.gather(1, ind)

    def predict(self, x):
        c5 = self.backbone(x)
        N = c5.shape[0]
        p5 = self.smooth(c5)
        p4 = self.deconv5(p5)
        p3 = self.deconv4(p4)
        p2 = self.deconv3(p3)
        cls_pred = self.cls_pred(p2)
        txty_pred = self.txty_pred(p2)
        twth_pred = self.twth_pred(p2)

        cls_pred = torch.sigmoid(cls_pred)
        # Find the 8-nearest neighbor maxima, where keep is the location of the hmax maxima
        # and cls_pred is the corresponding maxima
        hmax = nn.functional.max_pool2d(cls_pred, kernel_size=5, padding=2, stride=1)
        keep = (hmax == cls_pred).float()
        cls_pred *= keep
        # (N, 4, ws, hs) -> (N, ws, hs, 4) -> (N, ws * hs, 4)
        txtytwth_pred = (
            torch.cat([txty_pred, twth_pred], dim=1)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(N, -1, 4)
        )

        scale = (
            torch.tensor([[[448, 448, 448, 448]]])
            .float()
            .to(device=txtytwth_pred.device)
        )
        bbox_pred = torch.clamp((self.decode(txtytwth_pred) / scale)[0], 0.0, 1.0)

        topk_score, topk_ind, topk_clses = self.get_topk(cls_pred)
        topk_bbox_pred = bbox_pred[topk_ind[0]]
        return (
            topk_bbox_pred.detach().cpu().numpy(),
            topk_score[0].detach().cpu().numpy(),
            topk_clses[0].detach().cpu().numpy(),
        )


    def decode(self, pred):
        output = torch.zeros_like(pred)
        grid_y, grid_x = torch.meshgrid(
            [
                torch.arange(0, 112, device=pred.device),
                torch.arange(0, 112, device=pred.device),
            ], indexing='ij'
        )  # 448 / 4 = 112
        grid_cell = torch.stack([grid_x, grid_y], dim=-1).float().view(1, 112 * 112, 2)
        pred[:, :, :2] = (
            torch.sigmoid(pred[:, :, :2]) + grid_cell
        ) * 4  # Add offsets to grid centers
        pred[:, :, 2:] = (
            torch.exp(pred[:, :, 2:])
        ) * 4  # exp(log(w)), exp(log(h)) because we're predicting logs of width and height

        # Coordinate transformation: [cx, cy, w, h] -> [xmin, ymin, xmax, ymax]
        output[:, :, 0] = pred[:, :, 0] - pred[:, :, 2] / 2
        output[:, :, 1] = pred[:, :, 1] - pred[:, :, 3] / 2
        output[:, :, 2] = pred[:, :, 0] + pred[:, :, 2] / 2
        output[:, :, 3] = pred[:, :, 1] + pred[:, :, 3] / 2
        return output
