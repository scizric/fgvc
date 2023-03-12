# Data Augmentation
import torch
from torchvision import transforms

def give_transform():
  transform_rotate = [transforms.RandomRotation(45)]
  transform_blur = [transforms.GaussianBlur(5,(0.2,4))]
  transform_scale = [transforms.Resize(torch.randint(400,900,(1,1)).item()),
              transforms.CenterCrop(256)]
  transform_contrast = [transforms.RandomAutocontrast(0.8)]
  transform_erase = [transforms.ToTensor(),
                        transforms.RandomErasing(1,(0.01,0.1),(0.3,2.5)),
                        transforms.ToPILImage()
                      ]
  transform_distort=[transforms.RandomPerspective(0.5,1)]

  transform_normal = [transforms.RandomHorizontalFlip(p=0.5),
                      transforms.RandomRotation(10),
                      transforms.RandomAutocontrast(0.4),
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.ToTensor(),
                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                      ]

  transform=transforms.Compose(transform_normal)
  return transform
