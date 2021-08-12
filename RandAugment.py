import augmentation, random, torch
import torchvision.transforms as transforms


M = 30

color_range = torch.arange(0, 0.9+1e-8, (0.9-0)/M).tolist()
rotate_range = torch.arange(0, 30+1e-8, (30-0)/M).tolist()
shear_range = torch.arange(0, 0.3+1e-8, (0.3-0)/M).tolist()
translate_range = torch.arange(0, 10+1e-8, (10-0)/M).tolist()


Mag = {'Brightness' : color_range, 'Color' : color_range, 'Contrast' : color_range, 
       'Posterize' : torch.arange(4, 8+1e-8, (8-4)/M).tolist()[::-1], 'Sharpness' : color_range, 
       'Solarize' : torch.arange(0, 256+1e-8, (256-0)/M).tolist()[::-1], 'SolarizeAdd' : torch.arange(0, 110+1e-8, (110-0)/M).tolist(), 
       
       'Cutout' : torch.arange(0, 32*0.15+1e-8, (32*0.15-0)/M),
       
       'Rotate' : rotate_range, 'ShearX' : shear_range, 'ShearY' : shear_range,
       'TranslateX' : translate_range, 'TranslateY' : translate_range}


Fun = {'AutoContrast' : augmentation.AutoContrast, 'Brightness' : augmentation.Brightness, 
       'Color' : augmentation.Color, 'Contrast' : augmentation.Contrast,  'Equalize' : augmentation.Equalize, 'Invert' : augmentation.Invert, 
       'Posterize' : augmentation.Posterize, 'Sharpness' : augmentation.Sharpness, 
       'Solarize' : augmentation.Solarize, 'SolarizeAdd' : augmentation.SolarizeAdd,
       
       'Cutout' : augmentation.Cutout,
         
       'Rotate' : augmentation.Rotate, 'ShearX' : augmentation.ShearX, 'ShearY' : augmentation.ShearY, 
       'TranslateX' : augmentation.TranslateX, 'TranslateY' : augmentation.TranslateY}


transform_functions = list(Fun.keys())


class RandAug(torch.nn.Module):
    def __init__(self, N, M, p0, p1, pre_transform=[], post_transform=[], transform_functions=transform_functions):
        super().__init__()
        self.N = N
        self.M = M
        self.p0 = p0
        self.p1 = p1
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.transform_functions = transform_functions

    def forward(self, image):
        random_transform = self.pre_transform + RandAugment(self.N, self.M, self.p0, self.p1, self.transform_functions) + self.post_transform
        random_transform = transforms.Compose(random_transform)
        image = random_transform(image)
        return image
        
    def __repr__(self):
        return self.__class__.__name__ + "(N={0}, M={1}, p0={2}, p1={3})".format(self.N, self.M, self.p0, self.p1)


def RandAugment(N, M, p0, p1, transform_functions):
    '''
    N: number of augmentation transformations to apply
    M: magnitude for all transformations
    p0: probability of augmentations to apply
    p1: probability of functions to apply
    '''
    sampled_transforms = []
    if random.random() < p0:
        raw_sampled = random.choices(transform_functions, k=N)
        for transform in raw_sampled:
            if transform == 'Identity': continue
            elif transform in ['AutoContrast', 'Equalize', 'Invert']: sampled_transforms.append(Fun[transform](p1))
            else: sampled_transforms.append(Fun[transform](p1, Mag[transform][M]))
    return sampled_transforms