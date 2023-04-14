# Basic routines for trainning DL model
# Source: https://nextjournal.com/gkoehler/pytorch-mnist
# Import necessary files
import os
import torch
import tvault
import torch.distributed as dist
import torch.optim as optim
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from torch.nn.parallel import DistributedDataParallel as DDP

from module import MobileNetV2
import argparse

# seeding
seed = 2023
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = False
np.random.seed(seed)

# configurations
batch_size = 32
learning_rate = 0.1
log_interval = 1
mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


def train(model, train_epoch, train_loader, local_rank, criterion):
    model.train()
    for epoch in range(train_epoch):
        loss_acc = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(local_rank)
            target = target.to(local_rank)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss_acc += loss.item()
            loss.backward()
            optimizer.step()
        if epoch % log_interval == 0:
            print(f"Train Epoch: {epoch} \tLoss: {loss_acc / len(train_loader)}")


def test(model, test_loader, local_rank, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(local_rank)
            target = target.to(local_rank)
            output = model(data)
            test_loss += criterion(output, target).item()  # size avg?
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
    return 100.0 * correct / len(test_loader.dataset)


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--gpu_ids", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int, default=0)
    return parser


def init_for_distributed(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl", init_method="env://")
    if args.local_rank is not None:
        args.local_rank = local_rank
        print("Use GPU: {} for training".format(args.local_rank))
        torch.cuda.set_device(args.local_rank)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MNIST arg parser", parents=[get_args_parser()])
    args = parser.parse_args()

    # DDP
    init_for_distributed(args)

    # Model
    transform_train = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    # cifar100_training = CIFAR100Train(path, transform=transform_train)
    train_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    # cifar100_test = CIFAR100Test(path, transform=transform_test)
    test_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test
    )

    for batch_size in [32, 64, 128]:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
        )

        # for learning_rate in [0.01, 0.001]:
        model = MobileNetV2()
        print(f"start training for lr {learning_rate}")

        model = model.to(args.local_rank)
        model = DDP(model, device_ids=[args.local_rank])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        train(model, 40, train_loader, args.local_rank, criterion)
        if args.local_rank == 0:
            acc = test(model, test_loader, args.local_rank, criterion)
        tags = {
            "language": "pytorch",
            "size": "0.5x",
            "learning_rate": learning_rate,
            "epoch": 40,
            "batch_size": batch_size,
        }
        tvault.log_all(model, tags=tags, result=acc.item(), optimizer=optimizer)
