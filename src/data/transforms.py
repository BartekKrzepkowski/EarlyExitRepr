from torchvision.transforms import InterpolationMode
from torchvision import transforms

mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)

transform_blurred_train = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize(8, interpolation=InterpolationMode.BILINEAR, antialias=None), 
        transforms.Resize(32, interpolation=InterpolationMode.BILINEAR, antialias=None),
        transforms.RandomAffine(degrees=0, translate=(1/8, 1/8)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean, std)
    ])
transform_blurred_eval = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize(8, interpolation=InterpolationMode.BILINEAR, antialias=None), 
        transforms.Resize(32, interpolation=InterpolationMode.BILINEAR, antialias=None),
        transforms.Normalize(mean, std)
    ])
transform_proper_train = transforms.Compose([
        transforms.ToTensor(), 
        transforms.RandomAffine(degrees=0, translate=(1/8, 1/8)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean, std)
    ])
transform_proper_eval = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)
    ])


TRANSFORMS_NAME_MAP = {
    'transform_blurred_train': transform_blurred_train,
    'transform_blurred_eval': transform_blurred_eval,
    'transform_proper_train': transform_proper_train,
    'transform_proper_eval': transform_proper_eval,
}