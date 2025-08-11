from typing import Dict
from torch.utils.tensorboard import SummaryWriter

class TBoard(SummaryWriter):
    def __init__(self, log_path):
        super().__init__(log_dir=log_path)
        
        self.reset(0)

    def reset(self, start_step: int = 0):
        self.scalar_it = dict(train=start_step, val=start_step, test=start_step)

    def scalarit(self, maintag: str) -> int:
        assert maintag in ["train", "val", "test"], f"Expected maintag to be one of ['train', 'val', 'test']. Got: {maintag}"

        self.scalar_it[maintag] += 1
        return self.scalar_it[maintag]

    def add_multiple_scalars(self, maintag: str, scalars_dict: Dict[str, float]):
        assert maintag in ["train", "val", "test"], f"Expected tag to be one of ['train', 'val', 'test']. Got: {tag}"
    
        it = self.scalarit(maintag)
        
        # Remove this if you don't want val scalarit to match train scalarit
        if maintag == "val":
            it = self.scalar_it["train"]

        for tag, value in scalars_dict.items():
            self.add_scalar(f"{maintag.capitalize()}/{tag}", value, it)