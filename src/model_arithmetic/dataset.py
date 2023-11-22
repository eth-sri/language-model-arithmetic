import torch
from tqdm import tqdm
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset class for tokenized sequence data.

    Uses a tokenizer to convert text data from a DataFrame to input_ids (tokens), 
    and optionally attaches label data if present in the DataFrame.
    """
    def __init__(self, tokenizer, df, max_tokens=128, min_tokens=1, random_cutoff=False):
        """
        Initializes the CustomDataset object.

        Args:
            tokenizer (Tokenizer): The tokenizer to be used for the text data.
            df (pandas.DataFrame): DataFrame containing the text data, and optionally labels.
            max_tokens (int, optional): Maximum number of tokens per sequence. Defaults to 128.
            min_tokens (int, optional): Minimum number of tokens per sequence. Defaults to 1.
            random_cutoff (bool, optional): Whether to randomly cut off the number of tokens per sequence. Defaults to False.
        """
        super().__init__()
        data = df.dropna()
        self.tokenized_dataset = [
            tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids.view(-1) for sentence in tqdm(data["text"].tolist())
        ]

        self.df = data
        self.has_labels = "label" in data.columns
        self.min_tokens = min_tokens
        self.labels = None
        if self.has_labels:
            self.labels = data["label"].values
        
        self.random_cutoff = random_cutoff

    def __len__(self):
        """
        Returns the length of the tokenized dataset, 
        i.e., the number of tokenized sequences.
        
        Returns:
            int: Number of tokenized sequences.
        """
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        """
        Fetches an item from the dataset at the given index.

        If labels are available, also fetches the associated label.
        If `random_cutoff` is true, may truncate sequence length randomly.

        Args:
            idx (int): Index of the required sequence.

        Returns:
            dict: A dictionary with the following structure-
                {
                    "input_ids": torch.Tensor (Tokenized sequence),
                    "labels": torch.Tensor (Associated label, if available)
                }
        """
        cutoff = len(self.tokenized_dataset[idx])
        if self.random_cutoff:
            cutoff = torch.randint(min(cutoff, self.min_tokens), cutoff + 1, (1,)).item()
        
        if not self.has_labels:
            return {"input_ids": self.tokenized_dataset[idx][:cutoff]}
        else:
            return {"input_ids": self.tokenized_dataset[idx][:cutoff], "labels": torch.tensor([self.labels[idx]], dtype=torch.long)}
