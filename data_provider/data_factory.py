from data_provider.data_loader import Seq2Seq_DR_Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

data_dict = {
    "Huron": Seq2Seq_DR_Dataset,
    "Victoria": Seq2Seq_DR_Dataset
}

def data_provider(args, flag):

    Data = data_dict[args.data]

    data_path = f'./data/processed/{args.data}_5min_merged.csv'
    
    data_set = Data(
        data_path=data_path,
        flag=flag,
        time_window=args.time_window,
        output_size=args.output_size
    )
    
    shuffle_flag = True if flag == 'train' else False
    
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag
    )
    
    return data_set, data_loader
