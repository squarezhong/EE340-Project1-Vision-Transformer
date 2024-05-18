import datetime
import sys, getopt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from trainer import Trainer
from transformer import VisionTransformer

img_size = 28


def main(argv):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    vision_transformer = VisionTransformer()
    trainer = Trainer(vision_transformer, trainloader, testloader)
    trainer.train(epochs=50)
    trainer.test()


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main(sys.argv[1:])
    end_time = datetime.datetime.now()
    print('Time taken: {}'.format(end_time - start_time))