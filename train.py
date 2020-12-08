import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from data_loader import get_dataloader, get_transforms
from mask_keypoint_rcnn import maskkeypointrcnn_resnet34_fpn, maskkeypointrcnn_resnet50_fpn
from test import predict
from tqdm import tqdm
import os
import cv2
import time


class WarmUpLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warm_up_epochs, warm_up_factor):
        def f(epoch):
            if epoch >= warm_up_epochs:
                return 1
            alpha = epoch / warm_up_epochs
            return warm_up_factor * (1-alpha) + 1.0 * alpha
        super(WarmUpLR, self).__init__(optimizer, f)


def train_and_eval(model,
                   dataloaders,
                   epochs, batches_show=10,
                   lr=0.01,
                   keypoint_weight=1.0, mask_weight=1.0,
                   save_dir=None, save_interval=10,
                   load_from=None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("CUDA is unavailable, using CPU instead.")

    dataloader_train, dataloader_val = dataloaders
    val = False if dataloader_val is None else True

    model = model.to(device)
    if load_from is not None:
        model.load_state_dict(torch.load(load_from))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)

    model.train()
    for epoch in range(1, epochs+1):
        print("----------------------  TRAINING  ---------------------- ")
        running_losses = {}
        for i, data in enumerate(dataloader_train, start=1):
            images, targets = data
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss_dict["loss_keypoint"] = loss_dict["loss_keypoint"] * keypoint_weight
            loss_dict["loss_mask"] = loss_dict["loss_mask"] * mask_weight
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k, v in loss_dict.items():
                if k in running_losses:
                    running_losses[k] += v.item()
                else:
                    running_losses[k] = v.item()

            if i % batches_show == 0:
                running_losses = {k: v / batches_show for k, v in running_losses.items()}
                running_loss = sum(v for v in running_losses.values())
                summary = f'[epoch: {epoch}, batch: {i}] [loss: {running_loss:.3f}] ' + " ".join([f"[{k}: {v:.3f}]" for k, v in running_losses.items()])
                print(summary)
                running_losses = {}
        lr_scheduler.step(loss)

        if val:
            raise Exception("validation is not implemented yet")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            if epoch % save_interval == 0:
                print("----------------------   SAVING   ---------------------- ")
                # torch.save(model, os.path.join(save_dir, "epoch_{}.pth".format(epoch)))
                torch.save(model.state_dict(), os.path.join(save_dir, "epoch_{}.state_dict.pth".format(epoch)))


@ torch.no_grad()
def test_model(model, candidates_dir,
               kp_names=("head", "ass"),
               box_score_thre=0.5, kp_score_thre=0, mask_thre=0.5,
               save_image_out=True, show_image_out=False):
    out_dir = candidates_dir.rstrip("/") + ".detected"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    transforms = get_transforms(train=False)
    for p in tqdm(os.listdir(candidates_dir)):
        image_path = os.path.join(candidates_dir, p)
        image_cv2 = cv2.imread(image_path)

        canvas, results = predict(image_cv2, model,
                                  transforms=transforms, device=device, kp_names=kp_names,
                                  box_score_thre=box_score_thre, kp_score_thre=kp_score_thre, mask_thre=mask_thre,
                                  show=False)

        if save_image_out:
            cv2.imwrite(os.path.join(out_dir, p), canvas)
        if show_image_out:
            cv2.imshow("res", canvas)
            cv2.waitKey(0)
            cv2.destroyWindow("res")


if __name__ == '__main__':
    train_and_eval(
        model=maskkeypointrcnn_resnet50_fpn(num_classes=2, num_keypoints=2, pretrained=True),
        dataloaders=get_dataloader(img_dir="data/tmp_database/img/", lab_path="data/tmp_database/annotation.json",
                                   kp_names=("head", "ass"),
                                   train=True, batch_size=2, val_split=0.0, shuffle=True, num_workers=0),
        epochs=100, batches_show=1, save_dir="work_dir.test", save_interval=10,
        lr=0.001,
        keypoint_weight=0.1, mask_weight=1.0,
        load_from=None)

    model = maskkeypointrcnn_resnet50_fpn(num_classes=2, num_keypoints=2)
    model.load_state_dict(torch.load("work_dir.test/epoch_100.state_dict.pth"))
    test_model(model, "data/tmp_database/img",
               box_score_thre=0.5, kp_score_thre=0, mask_thre=0.5,
               save_image_out=True, show_image_out=False)
