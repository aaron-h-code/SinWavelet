import torch
import torch.nn.functional as F

import pdb


def rotation(x, augs):
    x_rot = []
    for aug in augs:
        if aug == 0:
            x_rot.append(x)
        elif aug == 90:
            # x_rot.append(x.transpose(2, 3).flip(2))
            x_rot.append(torch.rot90(x, k=1, dims=[3, 4]))
        elif aug == 180:
            # x_rot.append(x.flip(2).flip(3))
            x_rot.append(torch.rot90(x, k=2, dims=[3, 4]))
        elif aug == 270:
            # x_rot.append(x.transpose(2, 3).flip(3))
            x_rot.append(torch.rot90(x, k=3, dims=[3, 4]))

    return x_rot


def flipping(x, augs):
    x_flip = []
    for aug in augs:
        if aug == 'noflip':
            x_flip.append(x)
        elif aug == 'left-right':
            x_flip.append(x.flip(3))
        elif aug == 'bottom-up':
            x_flip.append(x.flip(2))
        elif aug == 'front-back':
            x_flip.append(x.flip(4))

    return x_flip


def cropping(x, augs):
    b, c, h, w, d = x.shape

    boxes = [[0, 0, h, w],
             [0, 0, h * 0.75, w * 0.75],
             [0, w * 0.25, h * 0.75, w],
             [h * 0.25, 0, h, w * 0.75],
             [h * 0.25, w * 0.25, h, w]]

    x_crop = []
    for aug in augs:
        i = augs.index(aug)
        cropped = x[:, :, int(boxes[i][0]):int(boxes[i][2]), int(boxes[i][1]):int(boxes[i][3]), :].clone()
        x_crop.append(F.interpolate(cropped, (h, w, d), mode='trilinear', align_corners=True))

    # for i in range(np.shape(boxes)[0]):
    #     cropped = x[:, :, int(boxes[i][0]):int(boxes[i][2]), int(boxes[i][1]):int(boxes[i][3])].clone()
    #     x_crop.append(F.interpolate(cropped, (h, w)))

    return x_crop


def augmenting_data(x, aug, aug_list):
    if aug == 'rotation':
        return rotation(x, aug_list)
    elif aug == 'flipping':
        return flipping(x, aug_list)
    elif aug == 'cropping':
        return cropping(x, aug_list)
    else:
        print('utils.augmenting_data: the augmentation type is not supported. Exiting ...')
        exit()
