from .models import CNNClassifier, save_model, load_model
from .utils import load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb
from torch.autograd import Variable
import torch.nn.functional as F


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()


def train(args):
    from os import path
    model = CNNClassifier()

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    DATA_PATH = 'dog breed data/images/all_images'
    LR = 0.1
    EPOCHS = 15

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, threshold=0.00001)
    model.train()
    model.cuda()
    data_train = load_data(DATA_PATH)

    global_step = 0
    for iteration in range(EPOCHS):
        print('iterations', iteration)
        print('LR', optimizer.param_groups[0]['lr'])
        for (batch_x, batch_y) in data_train:
            b_x = Variable(batch_x.cuda())
            b_y = Variable(batch_y.cuda())

            optimizer.zero_grad()

            o = model(b_x)
            loss = F.cross_entropy(o, b_y)

            train_logger.add_scalar('loss', loss, global_step=global_step)
            loss.backward()
            optimizer.step()
            global_step += 1
            scheduler.step(loss)
        if iteration % 10 == 0:
            save_model(model)
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
