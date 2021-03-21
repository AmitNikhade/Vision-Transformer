from __future__ import absolute_import, division, print_function
from tqdm import tqdm
import sys, os
from model import VIT
from Utils.preprocess import train_loader, test_loader
import torch
import argparse
import torch.optim as optim
import datetime, logging
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from Utils.config import parse_args


logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VIT.VIT().to(device)
logger.info("Parameters count: %s",sum(p.numel() for p in model.parameters()))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

logger.info("Training on: %s", device)
model.to(device)
logger.info("Now training")

if args.msp is None:
    logger.error("Model saving path should be passed")
    exit()
elif args.num_classes is None:
    logger.error("number of classes should be passed")
    exit()
elif args.train_data is None:
    logger.error("train data path should be passed") 
    exit()
elif args.test_data is None:
    logger.error("test data path should be passed")
    exit()
else:
    pass
def train():
    writer = SummaryWriter(log_dir=args.logs)
    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_accuracy = 0

    
        for data, label in tqdm(train_loader):
            data = data.float()
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            writer.add_scalar("train_loss", loss, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            writer.add_scalar("train_acc", acc, epoch)
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)
            writer.flush()
        with torch.no_grad():
            epoch_test_accuracy = 0
            epoch_test_loss = 0
            for data, label in test_loader:
                data = data.float()
                data = data.to(device)
                label = label.to(device)

                test_output = model(data)
                test_loss = criterion(test_output, label)
                writer.add_scalar("test_loss", loss, epoch)
                acc = (test_output.argmax(dim=1) == label).float().mean()
                writer.add_scalar("test_acc", acc, epoch)
                epoch_test_accuracy += acc / len(test_loader)
                epoch_test_loss += test_loss / len(test_loader)
                writer.flush()
                writer.close()
        
            print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_test_loss:.4f} - val_acc: {epoch_test_accuracy:.4f}\n"
        )
            model.train()

    torch.save(model,os.path.join(args.msp, 'model_weight_MNIST.pt'))
    logger.info("model saved to: data/")

if __name__ == "__main__":
    train()