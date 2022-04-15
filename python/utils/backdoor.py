import torch

def all2one_target_transform(x, attack_target=1):
    return torch.ones_like(x) * attack_target

def all2all_target_transform(x, num_classes):
    return (x + 1) % num_classes

def get_target_transform(args):
    """Get target transform function
    """
    if args.mode == 'all2one':
        target_transform = lambda x: all2one_target_transform(x, args.target_label)
    elif args.mode == 'all2all':
        target_transform = lambda x: all2all_target_transform(x, args.num_classes)
    else:
        raise Exception(f'Invalid mode {args.mode}')
    return target_transform
