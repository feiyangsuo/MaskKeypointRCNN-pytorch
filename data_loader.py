import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import random
import os
import cv2


class MaskKeypointDataset(Dataset):
    def __init__(self, img_dir, lab_path, transforms,
                 keypoint_names=("head", "ass"),
                 vis=False):
        self.vis = vis

        self.img_dir = img_dir
        self.transforms = transforms
        self.keypoint_names = keypoint_names
        self.num_keypoints = len(keypoint_names)

        self.img_fnames = []
        self.scatter_keypointss = []
        self.mask_polygonss = []
        with open(lab_path, "r") as f:
            labels = json.load(f)
            labels = list(labels.values())
        for lab in labels:
            crop_fname = lab["filename"]

            scatter_keypoints = [(reg["region_attributes"]["keypoint"], reg["shape_attributes"]["cx"], reg["shape_attributes"]["cy"])
                                 for reg in lab["regions"] if reg["shape_attributes"]["name"] == "point"]  # (kp_name, x, y)

            mask_polygons = [list(zip(reg["shape_attributes"]["all_points_x"], reg["shape_attributes"]["all_points_y"]))
                             for reg in lab["regions"] if reg["shape_attributes"]["name"] == "polygon"]

            self.img_fnames.append(crop_fname)
            self.scatter_keypointss.append(scatter_keypoints)
            self.mask_polygonss.append(mask_polygons)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_fnames[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        polygons = self.mask_polygonss[idx]
        scatter_kps = self.scatter_keypointss[idx]
        bboxes = []
        masks = []
        keypointss = []
        for polygon in polygons:
            bbox = (min([p[0] for p in polygon]), min([p[1] for p in polygon]),
                    max([p[0] for p in polygon]), max([p[1] for p in polygon]))  # (xmin, ymin, xmax, ymax)
            bboxes.append(bbox)

            mask = cv2.fillPoly(img=np.zeros_like(image[:, :, 0]), pts=[np.expand_dims(np.array(polygon), axis=1)], color=(1,))
            masks.append(mask)

            keypoints = []
            for kp_name in self.keypoint_names:
                candidates = [kp for kp in scatter_kps if kp[0] == kp_name]
                chosen = None
                for kp in candidates:
                    if mask[kp[2], kp[1]] == 1:
                        chosen = (kp[1], kp[2], 1)  # (x, y, visibility)
                        break
                if chosen is None:
                    chosen = [0, 0, 0]
                keypoints.append(chosen)
            keypointss.append(keypoints)
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        keypointss = torch.as_tensor(keypointss, dtype=torch.float32)

        image_id = torch.tensor([idx])
        labels = torch.ones((len(bboxes),), dtype=torch.int64)  # only have one class
        iscrowd = torch.zeros((len(bboxes),), dtype=torch.int64)  # all masks are not crowd

        target = {}
        target["image_id"] = image_id
        target["boxes"] = bboxes
        target["keypoints"] = keypointss
        target["masks"] = masks
        target["labels"] = labels
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        if self.vis:
            self.visualize(image, target)

        return image, target

    def __len__(self):
        return len(self.img_fnames)

    @ classmethod
    def collate_fn(cls, batch):
        return tuple(zip(*batch))  # Default collate_fn needs same size of images, so should re-define it here

    def visualize(self, image, target):
        image_cv2 = cv2.cvtColor((image.permute((1, 2, 0)).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        for bbox, keypoints, mask in zip(target["boxes"], target["keypoints"], target["masks"]):
            xmin, ymin, xmax, ymax = map(int, bbox)
            cv2.rectangle(image_cv2, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            image_cv2[:, :, 0] = np.where(mask.numpy() == 1, (255 + image_cv2[:, :, 0]) // 2, image_cv2[:, :, 0])
            for i, kp in enumerate(keypoints):
                x, y, visibility = map(int, kp)
                if visibility == 1:
                    cv2.circle(image_cv2, (x, y), 2, (0, 0, 255), 2)
                    cv2.putText(image_cv2, self.keypoint_names[i], (x, y-2), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        cv2.imshow("vis", image_cv2)
        cv2.waitKey(0)
        cv2.destroyWindow("vis")


def get_transforms(train):
    class ToTensor(object):
        def __call__(self, image, target):
            image = image.astype(np.float32) / 255
            image = torch.as_tensor(image).permute((2, 0, 1))
            return image, target

    class RandomHorizontalFlip(object):
        def __init__(self, prob):
            self.prob = prob

        def __call__(self, image, target):
            if random.random() < self.prob:
                height, width = image.shape[-2:]
                image = image.flip(-1)
                target["masks"] = target["masks"].flip(-1)

                bbox = target["boxes"]
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox
                keypoints = target["keypoints"]
                for keypoint in keypoints:
                    for kp in keypoint:
                        kp[0] = width - kp[0]
                target["keypoints"] = keypoints
            return image, target

    class Compose(object):
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, image, target):
            for t in self.transforms:
                image, target = t(image, target)
            return image, target

    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


def get_dataloader(img_dir, lab_path, kp_names,
                   train, batch_size, val_split=0.0, shuffle=True, num_workers=0):
    dataset = MaskKeypointDataset(img_dir=img_dir, lab_path=lab_path, keypoint_names=kp_names, transforms=get_transforms(train=train))
    if val_split == 0.0:
        dataset_train = dataset
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=MaskKeypointDataset.collate_fn)
        dataloader_val = None
    else:
        indices = torch.randperm(len(dataset)).tolist()
        where_train = int(len(dataset) * (1 - val_split))
        dataset_train = torch.utils.data.Subset(dataset, indices[:where_train])
        dataset_val = torch.utils.data.Subset(dataset, indices[where_train:])
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=MaskKeypointDataset.collate_fn)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=MaskKeypointDataset.collate_fn)
    return dataloader_train, dataloader_val


if __name__ == '__main__':
    dataset = MaskKeypointDataset(img_dir="data/tmp_database/img/",
                                  lab_path="data/tmp_database/annotation.json",
                                  keypoint_names=("head", "ass"),
                                  transforms=get_transforms(train=True),
                                  vis=True)
    it = iter(dataset)
    while True:
        try:
            a = next(it)
        except StopIteration:
            break
