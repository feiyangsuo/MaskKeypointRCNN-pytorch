import torch
from data_loader import get_transforms
from mask_keypoint_rcnn import maskkeypointrcnn_resnet34_fpn, maskkeypointrcnn_resnet50_fpn
import os
import cv2
import json
import time
import numpy as np


def load_model_eval(model_path, is_state_dict, use_cpu=False):
    device = torch.device("cuda") if (torch.cuda.is_available() and not use_cpu) else torch.device("cpu")
    if is_state_dict:
        model = maskkeypointrcnn_resnet50_fpn(num_classes=2, num_keypoints=6).to(device)
        model.load_state_dict(torch.load(model_path))
    else:
        model = torch.load(model_path).to(device)
    model.eval()
    transforms = get_transforms(train=False)
    return model, transforms, device


@ torch.no_grad()
def predict(image_cv2, model, transforms, device,
            kp_names=("head", "ass"),
            box_score_thre=0.5, kp_score_thre=0, mask_thre=0.5,
            show=False):
    image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    image, _ = transforms(image, None)
    image = [image.to(device)]
    pred = model(image)[0]

    canvas = np.copy(image_cv2)
    results = []
    for score, bbox, keypoints, keypoint_scores, mask in zip(pred["scores"], pred["boxes"], pred["keypoints"], pred["keypoints_scores"], pred["masks"]):
        if score >= box_score_thre:
            res = {"bbox": None, "mask": None, "keypoints": {kp_name: None for kp_name in kp_names}}
            xmin, ymin, xmax, ymax = map(int, bbox)
            cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            res["bbox"] = (xmin, ymin, xmax, ymax)

            mask = mask.squeeze().detach().cpu().numpy()
            canvas[:, :, 0] = np.where(mask >= mask_thre, (255 + canvas[:, :, 0]) // 2, canvas[:, :, 0])
            pos = np.where(mask >= mask_thre)
            res["mask"] = [(int(x), int(y)) for y, x in zip(pos[0], pos[1])]

            for i, (kp, kp_score) in enumerate(zip(keypoints, keypoint_scores)):
                x, y, visibility = map(int, kp)
                if visibility >= 0.5 and kp_score >= kp_score_thre:
                    cv2.circle(canvas, (x, y), 2, (0, 0, 255), 2)
                    cv2.putText(canvas, kp_names[i], (x, y - 2), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                    res["keypoints"][kp_names[i]] = (x, y)

            results.append(res)

    if show:
        cv2.imshow("res", canvas)
        cv2.waitKey(0)
        cv2.destroyWindow("res")

    return canvas, results
