from datasets import Dataset, load_dataset
from argparse import Namespace

class QAset:
    
    self.dataset = None
    
    @classmethod
    def load_data(cls, args: Namespace):
        self.dataset = load_dataset(args.origin_path, split=args.split)
        return self.dataset
    
    def save_with_config(self, args: Namespace):
        self.push_to_hub(args.output_path)
        print(f"Saved to {args.output_path}")
    
    def save_with_results(self, predictions, args: Namespace):
        pass

class NaturalQuestions(QAset):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, text):
        pass

class TriviaQA(QAset):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, text):
        pass

class HotpotQA(QAset):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, text):
        pass

class WebQuestions(QAset):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, text):
        pass

class BoolQ(QAset):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, text):
        pass

class StrategyQA(QAset):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, text):
        pass
