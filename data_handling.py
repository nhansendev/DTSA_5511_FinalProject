import torch
import numpy as np
import torchvision.transforms.v2 as v2

from constants import RANDOM_SEED


def identity(X):
    return X


def make_transforms():
    rotator = v2.RandomRotation(180, fill=-1)
    hflip = v2.RandomHorizontalFlip(p=1)
    vflip = v2.RandomVerticalFlip(p=1)
    orient_augmentor = v2.RandomChoice(
        transforms=[rotator, hflip, vflip, identity]
    ).cuda()

    resize_augmentor = v2.RandomChoice(
        transforms=[
            v2.RandomResizedCrop(size=(600, 600), scale=(0.7, 0.95)),
            identity,
        ]
    ).cuda()

    bjitter = v2.ColorJitter(brightness=(0.5, 1))
    sjitter = v2.ColorJitter(saturation=(0.5, 1))
    hjitter = v2.ColorJitter(hue=0.5)
    cjitter = v2.ColorJitter(contrast=(0.5, 1))
    color_augmentor = v2.RandomChoice(
        [bjitter, sjitter, hjitter, cjitter, identity]
    ).cuda()

    return orient_augmentor, resize_augmentor, color_augmentor


class DataHandler:
    def __init__(
        self,
        dataA,
        dataB,
        batch_size=4,
        use_geo_aug=True,
        use_crop_aug=True,
        use_color_aug=False,
        batch_aug=True,
        preallocate=False,
        seed=RANDOM_SEED,
    ) -> None:
        self.batch_size = batch_size
        self.use_geo_aug = use_geo_aug
        self.use_crop_aug = use_crop_aug
        self.use_color_aug = use_color_aug
        self.batch_aug = batch_aug
        self.seed = seed
        self.reset()

        self.dataA = (
            torch.permute(torch.FloatTensor(dataA) / 127.5 - 1, (0, 3, 1, 2))
            .half()
            .to("cuda" if preallocate else "cpu")
        )
        self.dataB = (
            torch.permute(torch.FloatTensor(dataB) / 127.5 - 1, (0, 3, 1, 2))
            .half()
            .to("cuda" if preallocate else "cpu")
        )

        self.geo_transform, self.crop_transform, self.color_transform = (
            make_transforms()
        )

        self.N_A = len(self.dataA)
        self.N_B = len(self.dataB)

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

    def make_sample(
        self,
        sample_size,
        use_A=True,
        use_geo_aug=None,
        use_crop_aug=None,
        use_color_aug=None,
    ):
        if use_geo_aug is None:
            use_geo_aug = self.use_geo_aug
        if use_crop_aug is None:
            use_crop_aug = self.use_crop_aug
        if use_color_aug is None:
            use_color_aug = self.use_color_aug

        idx = self.rng.choice(self.N_A if use_A else self.N_B, sample_size, False)

        return self.apply_augments(
            self.dataA[idx] if use_A else self.dataB[idx],
            use_geo_aug,
            use_crop_aug,
            use_color_aug,
            False,
        )

    def apply_augments(
        self, data, use_geo_aug, use_crop_aug, use_color_aug, batch_aug=None
    ):
        if batch_aug is None:
            batch_aug = self.batch_aug
        if batch_aug:
            if use_geo_aug:
                data = self.geo_transform(data)
            if use_crop_aug:
                data = self.crop_transform(data)
            if use_color_aug:
                data = self.color_transform(data * 0.5 + 0.5) * 2 - 1
            return data.half()
        else:
            N = data.shape[0]
            # Convert batch dimension to list
            data = [data[i] for i in range(N)]
            # Apply augments individually to maximize randomness
            # Half-precision not allowed by some transforms
            if use_geo_aug:
                for i in range(N):
                    data[i] = self.geo_transform(data[i].float())
            if use_crop_aug:
                for i in range(N):
                    data[i] = self.crop_transform(data[i].float())
            if use_color_aug:
                for i in range(N):
                    data[i] = self.color_transform(data[i].float() * 0.5 + 0.5) * 2 - 1
            return torch.stack(data).half()

    def make_batch(
        self,
        use_geo_aug=None,
        use_crop_aug=None,
        use_color_aug=None,
        batch_size=None,
        batch_aug=None,
    ):
        if use_geo_aug is None:
            use_geo_aug = self.use_geo_aug
        if use_crop_aug is None:
            use_crop_aug = self.use_crop_aug
        if use_color_aug is None:
            use_color_aug = self.use_color_aug
        if batch_size is None:
            batch_size = self.batch_size
        if batch_aug is None:
            batch_aug = self.batch_aug

        half_size = max(1, batch_size // 2)

        idxA = self.rng.choice(self.N_A, half_size, False)
        idxB = self.rng.choice(self.N_B, half_size, False)
        data = torch.cat([self.dataA[idxA], self.dataB[idxB]]).cuda()

        labels = (
            torch.FloatTensor([0] * half_size + [1] * half_size).unsqueeze(1).cuda()
        )

        data = self.apply_augments(
            data, use_geo_aug, use_crop_aug, use_color_aug, batch_aug=batch_aug
        )

        return data, labels


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dataA = rng.normal(size=(20, 600, 600, 3))

    DH = DataHandler(dataA, dataA)

    sample = DH.make_sample(2)
    print(sample.shape)

    data, labels = DH.make_batch()
    print(data.shape, labels.shape)
