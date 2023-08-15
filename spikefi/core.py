from .fault import Fault, FaultModel, FaultRound
from slayerSNN.slayer import _convLayer, _denseLayer, spikeLayer
from typing import List, Tuple
import torch
import copy
import random
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
import slayerSNN as snn

# TODO: Logging
# TODO: Documentation
# TODO: Order methods and fix their names


class Campaign:
    def __init__(self, net: torch.nn.Module, shape_in: Tuple[int]) -> None:
        self.golden_net = net
        self.faulty_net = copy.deepcopy(net)
        self.faulty_net.eval()
        self.device = next(net.parameters()).device

        self.slayer = [ch for ch in self.faulty_net.children() if type(ch) is spikeLayer][0]
        self.injectables = {name: lay
                            for name, lay in self.faulty_net.named_children()
                            if type(lay) is _convLayer or type(lay) is _denseLayer}
        self.layer_names = list(self.injectables.keys())

        self.shape_in = shape_in
        self.shapes_lay = {}
        self.shapes_wei = {}
        self.assume_shapes()

        self.rounds = [FaultRound(self.layer_names)]

    def __repr__(self) -> str:
        pass

    def inject(self, faults: List[Fault], round_idx: int = -1) -> List[Fault]:
        assert round_idx >= -len(self.rounds) and round_idx < len(self.rounds)

        inj_faults = self.assert_injected_faults(faults)
        self.define_random_faults(inj_faults)

        self.rounds[round_idx].insert(inj_faults)

        return inj_faults

    def then_inject(self, faults: List[Fault]) -> List[Fault]:
        self.rounds.append(FaultRound(self.layer_names))

        return self.inject(faults, -1)

    def eject(self, faults: List[Fault] = None, round_idx: int = None) -> None:
        # Eject all faults
        if not faults:
            self.rounds.clear()
            self.rounds.append(FaultRound(self.layer_names))
            return

        # Eject a group of faults, optionally from a specific round
        if round_idx is None:
            for r in self.rounds:
                r.remove(faults)
        else:
            self.rounds[round_idx].remove(faults)

    def register_hooks(self, round: FaultRound) -> List[RemovableHandle]:
        hook_handles = []
        for lay_name, h in round.hooks_map:
            lay_faults = round.get_faults(lay_name)
            if not lay_faults:
                continue

            lay = self.injectables[lay_name]
            if h[0]:
                handle = lay.register_forward_pre_hook(self.evaluation_pre_hook_wrapper(round))
                hook_handles.append(handle)
            if h[1]:
                handle = lay.register_forward_hook(self.evaluation_hook_wrapper(round))
                hook_handles.append(handle)

    def alter_synapses(self, round: FaultRound, action: str) -> None:
        a = action == 'perturb'
        z = action == 'restore'

        for lay_name, lay in self.injectables.items():
            for f in round.get_synapse_faults(lay_name):
                for s in f.sites:
                    if a:
                        # Perturb weights for synapse faults (no pre-hook needed)
                        lay.weight[s.unroll()] = f.model.perturb(lay.weight[s.unroll()])
                    elif z:
                        # Restore weights after the end of the round (no hook needed)
                        lay.weight[s.unroll()] = f.model.restore()

    def evaluate(self, round: FaultRound, test_loader: DataLoader) -> None:
        for b, (input, target, label) in enumerate(test_loader):
            input = input.to(self.device)
            target = target.to(self.device)

            output = self.faulty_net.forward(input)

            round.stats.testing.correctSamples += torch.sum(snn.predict.getClass(output) == label).item()
            round.stats.testing.numSamples += len(label)
            round.stats.print(0, b)
        round.stats.update()

    # TODO: Return results in a form of a statistics object
    def run(self, test_loader: DataLoader) -> None:
        for i, r in enumerate(self.rounds):
            pass

    def run_complete(self, fault_model: FaultModel, layers: List[str] = None) -> None:
        pass

    def assume_shapes(self) -> None:
        shape_hooks_handles = []
        for lay_name, lay in self.injectables.items():
            handle = lay.register_forward_hook(self.shape_hook_wrapper(lay_name))
            shape_hooks_handles.append(handle)

        dummy_input = torch.rand((1, *self.shape_in, 1)).to(self.device)
        self.faulty_net.forward(dummy_input)

        for h in shape_hooks_handles:
            h.remove()

    def define_random_faults(self, faults: List[Fault]) -> List[Fault]:
        for f in faults:
            for s in f.sites:
                if s.is_random():
                    s.layer = random.choice(self.layer_names)
                    s.set_chw(None)

                is_syn = f.model.is_synaptic()
                shapes = self.shapes_wei if is_syn else self.shapes_lay

                if s.dim0 is None:
                    s.dim0 = random.randrange(shapes[s.layer][0]) if is_syn else slice(None)

                new_chw = []
                for i, p in enumerate(s.get_chw()):
                    new_chw.append(p or random.randrange(shapes[s.layer][i+is_syn]))
                s.set_chw(tuple(new_chw))

        return faults

    def assert_injected_faults(self, faults: List[Fault]) -> List[Fault]:
        valid_faults = []
        for f in faults:
            fc = copy.deepcopy(f)
            v = True

            for s in fc.sites:
                v &= not s.layer or s.layer in self.injectables

                is_syn = fc.model.is_synaptic()
                if is_syn:
                    shapes = self.shapes_wei
                    v &= not s.dim0 or (s.dim0 >= 0 and s.dim0 < shapes[s.layer][0])
                else:
                    shapes = self.shapes_lay
                    # TODO: Check batch number (dim0)

                chw = s.get_chw()
                for i in range(3):
                    v &= not chw[i] or (chw[i] >= 0 and chw[i] < shapes[s.layer][i+is_syn])

                if not v:
                    fc.sites.remove(s)

            if bool(fc.sites):
                valid_faults.append(fc)

        return valid_faults

    def shape_hook_wrapper(self, layer_name: str):
        def shape_hook(module, args, output):
            self.shapes_lay[layer_name] = tuple(output.size()[1:4])
            self.shapes_wei[layer_name] = tuple(module.weight.data.size()[0:4])
        return shape_hook

    def infer_layer(self, out_size: torch.Size) -> torch.nn.Module:
        shape = tuple(out_size[1:4])

        try:
            lay_idx = list(self.shapes_lay.values()).index(shape)
            lay_name = list(self.shapes_lay.keys)[lay_idx]
            return self.injectables[lay_name]
        except ValueError:
            return None

    def evaluation_hook_wrapper(self, round: FaultRound):
        def evaluation_hook(slayer, args, output):
            pass
        return evaluation_hook

    def evaluation_pre_hook_wrapper(self, round: FaultRound):
        def evaluation_pre_hook(layer, args):
            pass
        return evaluation_pre_hook
