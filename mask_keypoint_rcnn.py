import torch
from torchvision.models.utils import load_state_dict_from_url
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNHeads, KeypointRCNNPredictor
import time


class MaskKeypointRCNN(FasterRCNN):
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 # Mask parameters
                 mask_roi_pool=None, mask_head=None, mask_predictor=None,
                 # keypoint parameters
                 keypoint_roi_pool = None, keypoint_head = None, keypoint_predictor = None,
                 num_keypoints = 17):

        out_channels = backbone.out_channels

        # mask predictor initialization
        assert isinstance(mask_roi_pool, (MultiScaleRoIAlign, type(None)))
        if num_classes is not None:
            if mask_predictor is not None:
                raise ValueError("num_classes should be None when mask_predictor is specified")
        if mask_roi_pool is None:
            mask_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=14,
                sampling_ratio=2)
        if mask_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)
        if mask_predictor is None:
            mask_predictor_in_channels = 256  # == mask_layers[-1]
            mask_dim_reduced = 256
            mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels,
                                               mask_dim_reduced, num_classes)

        # keypoint predictor initialization
        assert isinstance(keypoint_roi_pool, (MultiScaleRoIAlign, type(None)))
        if min_size is None:
            min_size = (640, 672, 704, 736, 768, 800)
        if num_classes is not None:
            if keypoint_predictor is not None:
                raise ValueError("num_classes should be None when keypoint_predictor is specified")
        if keypoint_roi_pool is None:
            keypoint_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=14,
                sampling_ratio=2)
        if keypoint_head is None:
            keypoint_layers = tuple(512 for _ in range(8))
            keypoint_head = KeypointRCNNHeads(out_channels, keypoint_layers)
        if keypoint_predictor is None:
            keypoint_dim_reduced = 512  # == keypoint_layers[-1]
            keypoint_predictor = KeypointRCNNPredictor(keypoint_dim_reduced, num_keypoints)

        super(MaskKeypointRCNN, self).__init__(
            backbone, num_classes,
            # transform parameters
            min_size, max_size,
            image_mean, image_std,
            # RPN-specific parameters
            rpn_anchor_generator, rpn_head,
            rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train, rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            # Box parameters
            box_roi_pool, box_head, box_predictor,
            box_score_thresh, box_nms_thresh, box_detections_per_img,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights)

        self.roi_heads.mask_roi_pool = mask_roi_pool
        self.roi_heads.mask_head = mask_head
        self.roi_heads.mask_predictor = mask_predictor
        self.roi_heads.keypoint_roi_pool = keypoint_roi_pool
        self.roi_heads.keypoint_head = keypoint_head
        self.roi_heads.keypoint_predictor = keypoint_predictor


model_urls = {
    'maskrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
    # legacy model for BC reasons, see https://github.com/pytorch/vision/issues/1606
    'keypointrcnn_resnet50_fpn_coco_legacy':
        'https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-9f466800.pth',
    'keypointrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth',
}


def maskkeypointrcnn_resnet50_fpn(pretrained=False, progress=True,
                                  num_classes=2, num_keypoints=7,
                                  pretrained_backbone=True, **kwargs):
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone)
    model = MaskKeypointRCNN(backbone, num_classes=num_classes, num_keypoints=num_keypoints, **kwargs)
    if pretrained:
        # TODO: merge parameters from pretrained maskrcnn and keypointrcnn
        # load mask_rcnn pretrained weights
        state_dict_mask = load_state_dict_from_url(model_urls['maskrcnn_resnet50_fpn_coco'], progress=progress)
        delete_predictor_weights(state_dict_mask)

        # load keypoint_rcnn pretrained weights
        state_dict_keypoint = load_state_dict_from_url(model_urls['keypointrcnn_resnet50_fpn_coco'], progress=progress)
        delete_predictor_weights(state_dict_keypoint)

        model.load_state_dict(state_dict_mask, strict=False)
        model.load_state_dict(state_dict_keypoint, strict=False)
    return model


def maskkeypointrcnn_resnet34_fpn(num_classes=2, num_keypoints=7,
                                  pretrained_backbone=True, **kwargs):
    backbone = resnet_fpn_backbone('resnet34', pretrained_backbone)
    model = MaskKeypointRCNN(backbone, num_classes=num_classes, num_keypoints=num_keypoints, **kwargs)
    return model


# The shape of detector's parameters doesn't match, since the number of classes differs
def delete_predictor_weights(state_dict, labels=("_predictor.",)):
    keys = list(state_dict.keys())
    for key in keys:
        for label in labels:
            if label in key:
                state_dict.pop(key)
                break


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mkrcnn = maskkeypointrcnn_resnet50_fpn(num_keypoints=3, pretrained=True).to(device)
    # test predicting
    mkrcnn.eval()
    img = [torch.rand(3, 214, 356).to(device)]
    for i in range(10):
        start = time.time()
        res = mkrcnn(img)
        end = time.time()
        print(f"predicting time: {(end - start)}s")
    # test training
    mkrcnn.train()
    img = [torch.rand(3, 214, 356).to(device)]
    boxes = torch.tensor([[ 81.,  47., 174., 345.],
                          [174.,  44., 252., 336.],
                          [251.,  80., 306., 236.],
                          [309.,  76., 367., 239.]], dtype=torch.float32)
    labels = torch.tensor([1, 1, 1, 1], dtype=torch.int64)
    masks = torch.randint(low=0, high=2, size=(4, 214, 356), dtype=torch.uint8)
    keypoints = torch.tensor([[[100, 100, 1], [0, 0, 0], [150, 200, 1]],
                              [[0, 0, 0], [0, 0, 0], [200, 50, 1]],
                              [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                              [[320, 200, 1], [330, 100, 1], [350, 150, 1]]], dtype=torch.float32)
    image_id = torch.tensor([1], dtype=torch.int64)
    area = torch.tensor([27714., 22776.,  8580.,  9454.], dtype=torch.float32)
    iscrowd = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
    target = {"boxes": boxes,
              "labels": labels,
              "masks": masks,
              "keypoints": keypoints,
              "image_id": image_id,
              "area": area,
              "iscrowd": iscrowd}
    target = [{k: v.to(device) for k, v in target.items()}]
    loss = mkrcnn(img, target)
    print(loss)
