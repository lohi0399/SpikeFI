import copy
import random
import types
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle

import slayerSNN as snn
from slayerSNN.slayer import _convLayer, _denseLayer, spikeLayer

from .fault import Fault, FaultModel, FaultSite, FaultRound

# TODO: Logging
# TODO: Documentation
# TODO: Order methods and fix their names


def _spike(self: spikeLayer, membrane_potential):
    return self(membrane_potential)


class Campaign:
    def __init__(self, net: torch.nn.Module, shape_in: Tuple[int], spike_layer: spikeLayer = None) -> None:
        self.golden_net = net
        self.faulty_net = copy.deepcopy(net)
        self.faulty_net.eval()
        self.device = next(net.parameters()).device

        if spike_layer:
            self.slayer = spike_layer
        else:
            try:
                self.slayer = [ch for ch in self.faulty_net.children() if type(ch) is spikeLayer][0]
            except IndexError:
                raise AssertionError("Spike layer not provided")

        self.slayer.forward = types.MethodType(snn.slayer.spikeLayer.spike, self.slayer)
        self.slayer.spike = types.MethodType(_spike, self.slayer)

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
        if not round.has_neuron_faults():
            return []

        handles = []
        # Neuron faults hook
        handles.append(self.slayer.register_forward_hook(self.neuron_hook_wrapper(round)))

        # Parametric faults hook
        for lay_name in self.layer_names[1:]:
            if round.has_parametric_faults(lay_name):
                handles.append(self.slayer.register_forward_hook(self.parametric_hook_wrapper(round)))
                break

        # 1st layer's parametric faults pre_hook
        if round.has_parametric_faults(self.layer_names[0]):
            handles.append(self.faulty_net.register_forward_pre_hook(self.parametric_pre_hook_wrapper(round)))

        return handles

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

    def create_dummy_layers(self, round: FaultRound) -> None:
        for f in round.get_parametric_faults():
            f.model.flayer = copy.deepcopy(self.slayer)
            f.model.flayer.neuron[f.model.param_name] = f.model.perturb_param(f.model.flayer.neuron[f.model.param_name])

    def evaluate(self, round: FaultRound, test_loader: DataLoader) -> None:
        for b, (input, target, label) in enumerate(test_loader):
            input = input.to(self.device)
            target = target.to(self.device)

            output = self.faulty_net.forward(input)

            round.stats.testing.correctSamples += torch.sum(snn.predict.getClass(output) == label).item()
            round.stats.testing.numSamples += len(label)
            round.stats.print(0, b)
        round.stats.update()

    def run(self, test_loader: DataLoader) -> None:
        for r in self.rounds:
            self.alter_synapses(r, 'perturb')
            handles = self.register_hooks(r)

            self.evaluate(r, test_loader)

            self.alter_synapses(r, 'restore')
            for h in handles:
                h.remove()

    def run_complete(self, test_loader: DataLoader, fault_model: FaultModel, layers: List[str] = None) -> None:
        self.rounds = []
        is_syn = fault_model.is_synaptic()
        shapes = self.shapes_wei if is_syn else self.shapes_lay

        for lay_name in layers or self.layer_names:
            lay_shape = shapes[lay_name]

            for k in range(lay_shape[0] if is_syn else 1):
                for c in range(lay_shape[0+is_syn]):
                    for h in range(lay_shape[1+is_syn]):
                        for w in range(lay_shape[2+is_syn]):
                            self.then_inject([Fault(fault_model, [FaultSite(lay_name, k if is_syn else None, (c, h, w))])])

        self.run(test_loader)

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
                    new_chw.append(p if p is not None else random.randrange(shapes[s.layer][i+is_syn]))
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

    def infer_layer(self, out_size: torch.Size) -> torch.nn.Module:
        shape = tuple(out_size[1:4])
        try:
            lay_idx = list(self.shapes_lay.values()).index(shape)
            lay_name = list(self.shapes_lay.keys())[lay_idx]
            return lay_name
        except ValueError:
            if shape == self.shape_in:
                return 'input'

            return None

    def parametric_perturbation(self, round: FaultRound, spikes: torch.Tensor) -> None:
        lay_name = self.infer_layer(spikes.shape)
        if not lay_name or lay_name == self.layer_names[-1]:
            return

        lay_idx = 0 if lay_name == 'input' else self.layer_names.index(lay_name)
        next_lay_name = self.layer_names[lay_idx]

        param_faults = round.get_parametric_faults(next_lay_name)
        if not param_faults:
            return

        next_lay = self.injectables[next_lay_name]
        for f in param_faults:
            f.model.perturb(f.model.flayer.spike(f.model.flayer.psp(next_lay(spikes)))[f.sites[0].unroll()])

    def shape_hook_wrapper(self, layer_name: str):
        def shape_hook(layer, args, output):
            self.shapes_lay[layer_name] = tuple(output.size()[1:4])
            self.shapes_wei[layer_name] = tuple(layer.weight.data.size()[0:4])
        return shape_hook

    def neuron_hook_wrapper(self, round: FaultRound):
        def neuron_hook(_, __, spikes_out):
            lay_name = self.infer_layer(spikes_out.shape)
            if not lay_name:
                return

            neu_faults = round.get_neuron_faults(lay_name)
            if not neu_faults:
                return

            for f in neu_faults:
                for s in f.sites:
                    spikes_out[s.unroll()] = f.model.perturbed or f.model.perturb(spikes_out[s.unroll()])

        return neuron_hook

    def parametric_hook_wrapper(self, round: FaultRound):
        def parametric_hook(_, __, spikes_out):
            self.parametric_perturbation(round, spikes_out)

        return parametric_hook

    def parametric_pre_hook_wrapper(self, round: FaultRound):
        def parametric_pre_hook(_, input):
            # TODO: Check if input is list of args
            self.parametric_perturbation(round, input)

        return parametric_pre_hook
