from collections.abc import Callable, Iterable, Iterator
from copy import deepcopy
import random
from threading import Lock, Thread
from time import sleep, time
from typing import Any, Optional

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle

import slayerSNN as snn
from slayerSNN.slayer import spikeLayer

from .fault import Fault, FaultModel, FaultRound, FaultSite


# FIXME: Fix long lines
# TODO: Logging
# TODO: Save results


class Campaign:
    def __init__(self, net: nn.Module, shape_in: tuple[int, int, int], slayer: spikeLayer) -> None:
        self.golden_net = net
        self.faulty_net = deepcopy(net)
        self.faulty_net.eval()
        self.device = next(net.parameters()).device
        self.slayer = slayer

        self.layers = {}
        self.names_lay = []
        self.names_inj = []
        self.shapes_lay = {}
        self.shapes_wei = {}
        self.injectables = {}
        self._assume_layer_info(shape_in)

        self.rounds = [FaultRound()]
        self.round_groups = {}

    def __repr__(self) -> str:
        s = 'FI Campaign:\n'
        s += f"  - Network: '{self.faulty_net.__class__.__name__}':\n"

        s += f"  - Layers ({len(self.layers)}): {{\n"
        for lay_idx, lay_name in enumerate(self.names_lay):
            has_weights = self.shapes_wei[lay_name][0] > 0
            neu = "{:2d} x {:2d} x {:2d}".format(*self.shapes_lay[lay_name])
            syn = "{:2d} x {:2d} x {:2d} x {:2d}".format(*self.shapes_wei[lay_name]) if has_weights else "-"

            s += f"      #{lay_idx:2d}: '{lay_name}' {type(self.layers[lay_name]).__name__} ({'' if lay_name in self.names_inj else 'non '}injectable)\n"
            s += f"        Shapes: neurons {neu} | synapses {syn}\n"
        s += '  }\n'

        s += f"  - Rounds ({len(self.rounds)}): {{\n"
        for round_idx, round in enumerate(self.rounds):
            round_str = str(round).replace('\n', '\n      ')
            s += f"      #{round_idx}: {round_str}\n"
        s += '  }'

        return s

    def _assume_layer_info(self, shape_in: tuple[int, int, int]) -> None:
        handles = []
        for name, child in self.faulty_net.named_children():
            h = child.register_forward_hook(self._layer_info_hook_wrapper(name))
            handles.append(h)

        dummy_input = torch.rand((1, *shape_in, 1)).to(self.device)
        self.faulty_net.forward(dummy_input)

        for h in handles:
            h.remove()

    def _layer_info_hook_wrapper(self, layer_name: str) -> Callable[[nn.Module, tuple[Any, ...], Tensor], None]:
        def layer_info_hook(layer: nn.Module, _, output: Tensor) -> None:
            # Unsupported layer types
            if isinstance(layer, snn.slayer._pspLayer) or isinstance(layer, snn.slayer._pspFilter) or isinstance(layer, snn.slayer._delayLayer):
                print('Attention: Unsupported layer type ' + type(layer) + ' found. Potential invalidity of results.')
                return

            has_weigths = isinstance(layer, nn.Conv3d) or isinstance(layer, nn.ConvTranspose3d)

            self.layers[layer_name] = layer
            self.names_lay.append(layer_name)

            self.shapes_lay[layer_name] = tuple(output.shape[1:4])
            self.shapes_wei[layer_name] = tuple(layer.weight.shape[0:4]) if has_weigths else (-1,) * 4

            if isinstance(layer, snn.slayer._convLayer) or isinstance(layer, snn.slayer._denseLayer):
                self.injectables[layer_name] = layer
                self.names_inj.append(layer_name)

        return layer_info_hook

    def inject(self, faults: Iterable[Fault], round_idx: int = -1) -> list[Fault]:
        assert -len(self.rounds) <= round_idx < len(self.rounds), f'Invalid round index {round_idx}'

        inj_faults = self.validate(faults)
        self.define_random(inj_faults)

        self.rounds[round_idx].insert_many(inj_faults)

        return inj_faults

    def validate(self, faults: Iterable[Fault]) -> list[Fault]:
        if not isinstance(faults, Iterable):
            raise TypeError(f"'{type(faults).__name__}' object is not iterable")

        valid_faults = []
        for f in faults:
            is_syn = f.model.is_synaptic()
            shapes = self.shapes_wei if is_syn else self.shapes_lay

            if f.model.is_parametric() and f.model.param_name not in self.slayer.neuron:
                continue

            to_remove = set()
            for s in f.sites:  # Validate only the defined fault sites
                v = s.layer in self.injectables
                if v:
                    if is_syn:
                        v &= -shapes[s.layer][0] <= s.position[0] < shapes[s.layer][0]

                    for i in range(1, 4):
                        # shapes_lay index values, si: 0-2
                        # shapes_wei index values, si: 1-3
                        si = i - (not is_syn)
                        v &= -shapes[s.layer][si] <= s.position[i] < shapes[s.layer][si]

                if not v:
                    to_remove.add(s)

            f.sites.difference_update(to_remove)
            if f:
                valid_faults.append(f)

        return valid_faults

    def define_random(self, faults: Iterable[Fault]) -> Iterable[Fault]:
        if not isinstance(faults, Iterable):
            raise TypeError(f"'{type(faults).__name__}' object is not iterable")

        has_site_duplicates = False
        for f in faults:
            is_syn = f.model.is_synaptic()
            shapes = self.shapes_wei if is_syn else self.shapes_lay

            for s in f.sites_pending:
                if not s.layer:
                    s.layer = random.choice(self.names_inj)

                pos = list(s.position)
                if pos[0] is None:
                    pos[0] = random.randrange(shapes[s.layer][0]) if is_syn else slice(None)

                for i in range(1, 4):
                    if pos[i] is None:
                        si = i - (not is_syn)
                        pos[i] = random.randrange(shapes[s.layer][si])

                s.position = tuple(pos)

            f.refresh(discard_duplicates=False)
            has_site_duplicates |= bool(f.sites_pending)

        if has_site_duplicates:
            print('Some of the newly defined random fault sites already exist.')

        return faults

    def then_inject(self, faults: Iterable[Fault]) -> list[Fault]:
        self.rounds.append(FaultRound())
        return self.inject(faults, -1)

    def eject(self, faults: Iterable[Fault] = None, round_idx: int = None) -> None:
        if faults is not None and not isinstance(faults, Iterable):
            raise TypeError(f"'{type(faults).__name__}' object is not iterable")

        # Eject from a specific round
        if round_idx:
            # Eject indicated faults from the round
            if faults:
                self.rounds[round_idx].extract_many(faults)
            # Eject all faults from the round, i.e., remove the round itself
            if not faults or not self.rounds[round_idx]:
                self.rounds.pop(round_idx)
        # Eject from all rounds
        else:
            # Eject indicated faults from any round the might exist
            if faults:
                for r in self.rounds:
                    r.extract_many(faults)
                    if not r:
                        self.rounds.pop(r)
            # Eject all faults from all rounds, i.e., all the rounds themselves
            else:
                self.rounds.clear()

        if not self.rounds:
            self.rounds.append(FaultRound())

    def run(self, test_loader: DataLoader, error: snn.loss = None, forward: Callable[[Tensor, Optional[str], Optional[str]], Iterator[Tensor]] = None) -> None:
        self._prepare_rounds()
        forward_ = forward or self._forward

        self.progress = Progress(len(test_loader), len(self.rounds))

        progr_thread = Thread(target=self._refresh_progress_job, args=(.001,))
        progr_thread.daemon = True
        progr_thread.start()
        progr_lock = Lock()

        # TODO: Check fault-free inference
        for b, (input, target, label) in enumerate(test_loader):  # For each batch
            spikes_in = input.to(self.device)
            layer_sta = None

            for group_lay_name, round_group in self.round_groups.items():  # For each fault round group
                spikes_in_iter = forward_(spikes_in, layer_sta, group_lay_name)
                for spikes_in in spikes_in_iter:
                    pass

                spikes_out_iter = forward_(spikes_in, group_lay_name)
                spikes_out = next(spikes_out_iter)
                layer_sta = group_lay_name

                is_out_faulty = group_lay_name == self.names_lay[-1]

                for r in round_group:  # For each fault round
                    round = self.rounds[r]
                    handles = self._register_hooks(round)  # TODO: Register once and separate by fault round

                    self._perturb_synap(round)
                    self._perturb_param(round)

                    round_spikes_iter = forward_(spikes_in, group_lay_name)

                    output = next(round_spikes_iter)
                    early_stop = not error and not is_out_faulty and torch.equal(output, spikes_out)

                    if not early_stop:
                        for output in round_spikes_iter:
                            pass
                        if is_out_faulty:
                            out_neuron_callable = self._neuron_pre_hook_wrapper(round, self.names_lay[-1])
                            with torch.no_grad():
                                out_neuron_callable(self.layers[self.names_lay[-1]], (output,))

                    # Testing stats
                    # FIXME: Implement early stop correctly
                    round.stats.testing.correctSamples += len(label) if early_stop else torch.sum(snn.predict.getClass(output) == label).item()
                    round.stats.testing.numSamples += len(label)
                    if error:
                        round.stats.testing.lossSum += error.numSpikes(output, target.to(self.device)).cpu().item()

                    self._restore_param(round)
                    self._restore_synap(round)

                    for h in handles:
                        h.remove()

                    with progr_lock:
                        self.progress.step()

            with progr_lock:
                self.progress.step_batch()

        for round in self.rounds:
            round.stats.update()

    def _prepare_rounds(self) -> None:
        for r, round in enumerate(self.rounds):
            round = FaultRound(sorted(round.items(), key=lambda item: self.names_lay.index(item[0][0])))
            min_lay_name = next(iter(round.keys()), 'nominal')[0]

            self.round_groups.setdefault(min_lay_name, list())
            self.round_groups[min_lay_name].append(r)

        self.round_groups = dict(sorted(self.round_groups.items(), key=lambda item: self.names_lay.index(item[0])))

    def _forward(self, spikes_in: Tensor, layer_start: str = None, layer_end: str = None) -> Iterator[Tensor]:
        s = torch.clone(spikes_in)
        has_started = not layer_start

        for lay_name in self.names_lay:
            if lay_name == layer_end:
                break
            has_started |= lay_name == layer_start
            if not has_started:
                continue

            if not isinstance(self.layers[lay_name], snn.slayer._dropoutLayer):
                with torch.no_grad():
                    s = self.slayer.spike(self.slayer.psp(self.layers[lay_name](s)))

            yield s

    def _register_hooks(self, round: FaultRound) -> list[RemovableHandle]:
        faults = round.search_neuronal()
        if not faults:
            return []

        handles = []
        for f in faults:
            is_param = f.model.is_parametric()
            for s in f.sites:
                lay_idx = self.names_lay.index(s.layer)

                # Neuron fault pre-hooks
                # Neuron faults for last layer are evaluated directly on network's output
                if lay_idx < len(self.layers) - 1:
                    next_lay = self.layers[self.names_lay[lay_idx + 1]]
                    h = next_lay.register_forward_pre_hook(self._neuron_pre_hook_wrapper(round, s.layer))
                    handles.append(h)

                # Parametric fault hooks
                if is_param:
                    h = self.layers[s.layer].register_forward_hook(self._parametric_hook_wrapper(round, s.layer))
                    handles.append(h)

        return handles

    def _perturb_synap(self, round: FaultRound) -> None:
        for f in round.search_synaptic():
            for s in f.sites:
                # Perturb weights for synapse faults (no pre-hook needed)
                lay = self.injectables[s.layer]
                with torch.no_grad():
                    lay.weight[s.unroll()] = f.model.perturb(lay.weight[s.unroll()], s)

    def _restore_synap(self, round: FaultRound) -> None:
        for f in round.search_synaptic():
            for s in f.sites:
                # Restore weights after the end of the round (no hook needed)
                original = f.model.restore(s)
                if original is not None:
                    with torch.no_grad():
                        self.injectables[s.layer].weight[s.unroll()] = original

    def _perturb_param(self, round: FaultRound) -> None:
        # Create a dummy layer for each parametric fault
        for f in round.search_parametric():
            f.model.param_perturb(self.slayer)

    def _restore_param(self, round: FaultRound) -> None:
        # Destroy dummy layers
        for f in round.search_parametric():
            f.model.param_restore()

    def show_progress(self) -> None:
        print('\033[1A\x1b[2K' * self.progress._flush_lines_num)  # Line up, line clear
        print(self.progress)

    def _refresh_progress_job(self, period_secs: float) -> None:
        while self.progress.iter < self.progress.iter_num:
            self.show_progress()
            sleep(period_secs)

        self.show_progress()

    def run_complete(self, test_loader: DataLoader, fault_model: FaultModel, layer_names: Iterable[str] = None, error: snn.loss = None) -> None:
        if not isinstance(layer_names, Iterable) or isinstance(layer_names, str):
            raise TypeError(f"'{type(layer_names).__name__}' object for layer_names arguement is not iterable or is str")

        self.rounds.clear()

        is_syn = fault_model.is_synaptic()
        shapes = self.shapes_wei if is_syn else self.shapes_lay

        lay_names_inj = [lay_name for lay_name in layer_names if lay_name in self.names_inj] if layer_names else self.names_inj

        for lay_name in lay_names_inj:
            lay_shape = shapes[lay_name]
            for k in range(lay_shape[0] if is_syn else 1):
                for l in range(lay_shape[0 + is_syn]):          # noqa: E741
                    for m in range(lay_shape[1 + is_syn]):
                        for n in range(lay_shape[2 + is_syn]):
                            self.then_inject(
                                [Fault(fault_model, FaultSite(lay_name, (k if is_syn else slice(None), l, m, n)))])

        self.run(test_loader, error)

    def _neuron_pre_hook_wrapper(self, round: FaultRound, prev_layer_name: str) -> Callable[[nn.Module, tuple[Any, ...]], None]:
        def neuron_pre_hook(_, args: tuple[Any, ...]) -> None:
            prev_spikes_out = args[0]
            for f in round.search_neuronal(prev_layer_name):
                for s in f.sites:
                    prev_spikes_out[s.unroll()] = f.model.perturb(prev_spikes_out[s.unroll()], s)

        return neuron_pre_hook

    def _parametric_hook_wrapper(self, round: FaultRound, layer_name: str) -> Callable[[nn.Module, tuple[Any, ...]], None]:
        def parametric_hook(layer: nn.Module, _, spikes_out: Tensor) -> None:
            for f in round.search_parametric(layer_name):
                flayer = f.model.flayer
                fspike_out = flayer.spike(flayer.psp(spikes_out))
                for s in f.sites:
                    f.model.args[0][s] = fspike_out[s.unroll()]

        return parametric_hook


class Progress:
    def __init__(self, batches_num: int, rounds_num: int) -> None:
        self.status = 0.
        self.batch = 0
        self.batch_num = batches_num
        self.iter = 0
        self.iter_num = batches_num * rounds_num
        self.fragment = 1. / self.iter_num
        self.start_time = time()
        
        self._flush_lines_num = 0

    def __str__(self) -> str:
        s = " Batch #\tTotal time\tProgress\n"
        s += f"{self.batch + 1:4d}/{self.batch_num:d}\t"
        s += f"{(time() - self.start_time):.3f} sec\t"
        s += f"{self.status * 100.:6.2f} %\t\n"

        self._flush_lines_num = s.count('\n') + 2
        
        return s

    def step(self) -> None:
        self.status += self.fragment
        self.iter += 1
    
    def step_batch(self) -> None:
        self.batch += 1
