"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils
import argparse 
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description="train on pizza_steak_sushi")
    
    parser.add_argument("--learning_rate", 
                        type=float, 
                        default=0.001)
    
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=32)
    
    parser.add_argument("--num_epochs", 
                        type=int, 
                        default=10)
    
    parser.add_argument("--hidden_units", 
                        type=int, 
                        default=10)
    
    return parser.parse_args()


args = parse_args()

NUM_EPOCHS = args.num_epochs      
BATCH_SIZE = args.batch_size       
HIDDEN_UNITS = args.hidden_units   
LEARNING_RATE = args.learning_rate

train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

device = "mps" if torch.backends.mps.is_available() else "cpu"

data_transform = transforms.Compose([transforms.Resize((64, 64)),
                                     transforms.ToTensor()])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
    )

model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
    ).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

utils.save_model(model=model,
                target_dir="models",
                model_name="05_going_modual_tinyvgg_model.pth")
