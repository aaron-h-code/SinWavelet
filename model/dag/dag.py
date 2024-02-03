import torch

from .utils import augmenting_data
from . import config as config

import pdb


class DAG(object):
    def __init__(self, D_loss_func, G_loss_func, policy=['flipping'], policy_weight=[1.0]):
        print('Initializing DAG ...')
        self.D_loss_func = D_loss_func
        self.G_loss_func = G_loss_func
        self.policy = policy
        self.policy_weight = policy_weight

    def get_num_of_augments_from_policy(self):
        num_of_augments = 0
        for i in range(len(self.policy)):
            num_of_augments += len(config.augment_list[self.policy[i]])

        return num_of_augments

    def get_augmented_samples_from_policy(self, x):
        x_arg = []
        for aug_type in self.policy:
            x_arg.append(augmenting_data(x, aug_type, config.augment_list[aug_type]))
        x_arg = torch.cat(x_arg, 0)

        return x_arg

    def compute_discriminator_loss(self, x_real, x_fake, netD):
        ''' compute D loss for original augmented real/fake data samples '''

        loss_r_aug = 0
        loss_f_aug = 0
        gradient_penalty_aug = 0
        loss_aug = 0
        aug_idx = 0
        for i in range(len(self.policy)):
            x_real_aug = augmenting_data(x_real, self.policy[i], config.augment_list[self.policy[i]])
            x_fake_aug = augmenting_data(x_fake, self.policy[i], config.augment_list[self.policy[i]])
            aug_w = self.policy_weight[i]
            n_aug_type = len(config.augment_list[self.policy[i]])
            for j in range(n_aug_type):
                loss_r, loss_f, gradient_penalty, loss = self.D_loss_func(x_real_aug[j], x_fake_aug[j], netD, dag=True, dag_idx=aug_idx + j)
                loss_r_aug = loss_r_aug + aug_w * loss_r / n_aug_type
                loss_f_aug = loss_f_aug + aug_w * loss_f / n_aug_type
                gradient_penalty_aug = gradient_penalty_aug + aug_w * gradient_penalty / n_aug_type
                loss_aug = loss_aug + aug_w * loss / n_aug_type
            aug_idx += n_aug_type

        return loss_r_aug, loss_f_aug, gradient_penalty_aug, loss_aug

    def compute_generator_loss(self, x_fake, netD):
        ''' compute G loss for original augmented real/fake data samples '''

        loss_adv_aug = 0
        loss_aug = 0
        aug_idx = 0
        for i in range(len(self.policy)):
            x_fake_aug = augmenting_data(x_fake, self.policy[i], config.augment_list[self.policy[i]])
            aug_w = self.policy_weight[i]
            n_aug_type = len(config.augment_list[self.policy[i]])
            for j in range(n_aug_type):
                loss_adv, loss = self.G_loss_func(x_fake_aug[j], netD, dag=True, dag_idx=aug_idx + j)
                loss_adv_aug = loss_adv_aug + aug_w * loss_adv / n_aug_type
                loss_aug = loss_aug + aug_w * loss / n_aug_type
            aug_idx += n_aug_type

        return loss_adv_aug, loss_aug


if __name__ == "__main__":

    import torchvision
    import torchvision.utils as vutils
    import torchvision.transforms as transforms

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=6, drop_last=True)

    policy = ['cropping']
    dag = DAG(None, None, policy=policy)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs_aug = dag.get_augmented_samples(inputs)
        for i in range(len(inputs_aug)):
            vutils.save_image(torch.tensor(inputs_aug[i]), 'output_{}_{}.png'.format(policy[0], i), normalize=True,
                              scale_each=True, nrow=10)
        break
