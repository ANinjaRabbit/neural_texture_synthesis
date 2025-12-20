import torch
import torch.nn as nn
import cv2
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from tqdm import tqdm
imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
imagenet_std = torch.tensor([0.229, 0.224, 0.225])
inv_imagenet_mean = -imagenet_mean / imagenet_std
inv_imagenet_std = 1.0 / imagenet_std
forward_transform = transforms.Compose([
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

inv_transform = transforms.Compose([
    transforms.Normalize(mean=inv_imagenet_mean, std=inv_imagenet_std),
])



def calculate_gram(input):

    # N C H W -> N C C
    assert len(input.shape)==4

    batch_size,channels,H,W=input.shape
    feature=input.view(batch_size , channels,H*W)
    G = torch.matmul(feature, feature.transpose(1, 2))
    G.div_(H*W)
    return G

def gram_mse_loss(syn,gt , device):
    # [ tensor(N , C , C)]
    total_loss = torch.tensor(0.0).to(device)
    for i in range(len(syn)):
        total_loss +=  torch.mean((syn[i]-gt[i])**2)
    return total_loss


def read_image(path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    image = image.unsqueeze(0)

    return image

def save_image(tensor, file_name):
    tensor = inv_transform(tensor)
    tensor = torch.clamp(tensor,0,1)
    img_pil = TF.to_pil_image(tensor,mode='RGB')
    img_pil.save(file_name)



def synthesize_texture(model,gt, save_path ,  device , epoch = 2000 , lr = 0.01 , optimizer = 'LBFGS'):


    model.to(device)
    gt = gt.to(device)

    syn = torch.randn(gt.shape)
    syn = syn.to(device).requires_grad_(True)

    model(gt)
    gt_grams = [calculate_gram(fmap) for fmap in model.feature_maps]

    if optimizer == 'Adam':
        optimizer = torch.optim.Adam([syn], lr=lr)
        for i in tqdm(range(epoch)):
            optimizer.zero_grad()
            model(syn)
            syn_grams = [calculate_gram(fmap) for fmap in model.feature_maps]
            loss = gram_mse_loss(syn_grams,gt_grams , device)
            loss.backward(retain_graph=True)

            optimizer.step()


            if (i+1)%100 == 0:
                save_image(syn.squeeze(0), "images/epochs/epoch_{}.jpg".format(i+1))
    elif optimizer == 'LBFGS':
        optimizer = torch.optim.LBFGS([syn], lr=lr)

        def closure():
            optimizer.zero_grad()
            model(syn)
            syn_grams = [calculate_gram(fmap) for fmap in model.feature_maps]
            loss = gram_mse_loss(syn_grams,gt_grams , device)
            loss.backward(retain_graph=True)

            return loss
        
        for i in tqdm(range(epoch)):

            loss = optimizer.step(closure)


            if (i+1)%100 == 0:
                print(f"loss: {loss.item()}")
                save_image(syn.squeeze(0), "images/epochs/epoch_{}.jpg".format(i+1))
    else:
        raise NotImplementedError

    
    

    save_image(syn.squeeze(0), save_path)

