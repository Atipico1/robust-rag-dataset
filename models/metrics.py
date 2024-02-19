from dataclasses import dataclass

@dataclass
class Metrics:
    em: float
    em_ans: float
    em_unans: float
    em_conflict: float
    em_nonconflict: float
    em_adversary: float
    em_nonadversary: float
    f1: float
    hasanswer: float
    acc: float
    acc_ans: float
    acc_unans: float
    acc_conflict: float
    acc_nonconflict: float
    acc_adversary: float
    acc_nonadversary: float

    def __str__(self):
        return f'Accuracy: {self.accuracy:.2f}, Precision: {self.precision:.2f}, Recall: {self.recall:.2f}, F1: {self.f1:.2f}'