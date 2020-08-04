from torch.utils.data import DataLoader

from siim_pneumothorax.dataset.pneumo_dataset import PneumoDataset

def get_pneumo_loaders(df, img_size, is_train, grid_size, 
    rgb_channel, batch_size, shuffle, num_workers):
    # Create the dataset and return the dataloader for torch usage
    dataset = PneumoDataset(
        df=df, rgb_channel=rgb_channel,
        img_size=img_size, grid_size=grid_size, is_train=is_train
    )

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )