from dataloaders.datasets import vaihingen , landslide
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):
    if args.dataset == 'vaihingen':
        train_set = vaihingen.Vaihingen(args, split='train')
        val_set = vaihingen.Vaihingen(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        print(train_loader)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        print(val_loader)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'landslide':
        train_set = landslide.Landslide(args, split='train')
        val_set = landslide.Landslide(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        print(train_loader)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True, **kwargs)
        print(val_loader)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    else:
        raise NotImplementedError

