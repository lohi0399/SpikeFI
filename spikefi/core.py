from . import fault
from typing import List


class Campaign:
    def __init__(self, net) -> None:
        self.golden_net = net

    def inject(self, faults: List[fault.Fault]):
        pass

    def eject():
        pass

    def run():
        pass
