# author_='bo.li';
# date: 3/21/22 2:50 PM
import os
import sys
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, datasets, utils
from torchcontrib.optim import SWA
import argparse
from model import AlexNet
import sys
sys.path.append("../")
from util import count_parameters, load_cfg_from_yaml
from trainer import CNNTrainer

# init transform
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}


def get_args():
    parser = argparse.ArgumentParser(description='AlexNet')
    parser.add_argument('--config', type=str, default='./configs/AlexNet.yaml')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # set rand seed
    torch.manual_seed(2)
    net = AlexNet(num_classes=5, init_weights=True)
    # load device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    args = get_args()
    configs = load_cfg_from_yaml(args.config)

    image_path = configs['DATASET']['data_dir']  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), configs['TRAIN']['bs'] if configs['TRAIN']['bs'] > 1 else 0, 8])  # number of workers
    print(f"Using {nw} dataloader workers every process")

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=configs['TRAIN']['bs'], shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=configs['TRAIN']['bs'], shuffle=False,
                                                  num_workers=nw)
    # show images
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    # optimizer
    optimizer = eval('torch.optim.' + configs['TRAIN']['opt'])(net.parameters(), configs['TRAIN']['lr'])
    optimizer = SWA(optimizer)

    # loss function
    loss_func = torch.nn.CrossEntropyLoss()

    alexnettrainer = CNNTrainer(model_save_dir=configs['TRAIN']['weight_save_path'], log_dir=configs['TRAIN']['log_dir'])
    alexnettrainer.fit(model=net,
                       train_loader=train_loader,
                       validate_loader=validate_loader,
                       num_classes=configs['TRAIN']['num_classes'],
                       optimizer=optimizer,
                       criterions=loss_func,
                       epochs=configs['TRAIN']['epochs'],
                       model_name=configs['Network']['name']
                       )
