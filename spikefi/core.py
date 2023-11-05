from collections.abc import Callable, Iterable, Iterator
from copy import deepcopy
import random
from threading import Lock, Thread
from typing import Any

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle

import slayerSNN as snn
from slayerSNN.slayer import spikeLayer

from .fault import Fault, FaultModel, FaultRound, FaultSite
from .utils.layer import LayersInfo
from .utils.progress import CampaignProgress, refresh_progress_job


# FIXME: Fix long lines
# TODO: Logging + silent
# TODO: Save results


class Campaign:
    def __init__(self, net: nn.Module, shape_in: tuple[int, int, int], slayer: spikeLayer) -> None:
        self.golden_net = net
        self.device = next(net.parameters()).device
        self.slayer = slayer

        self.layers_info = LayersInfo()
        self.layers = {}  # TODO: Remove
        self.assume_layers_info(shape_in)

        self.rounds = [FaultRound()]
        self.round_groups: dict[str, list[int]] = {}

    def __repr__(self) -> str:
        s = 'FI Campaign:\n'
        s += f"  - Network: '{self.golden_net.__class__.__name__}':\n"
        s += f"  - {str(self.layers_info).replace('}', '  }')}\n"
        s += f"  - Rounds ({len(self.rounds)}): {{\n"
        for round_idx, round in enumerate(self.rounds):
            round_str = str(round).replace('\n', '\n      ')
            s += f"      #{round_idx}: {round_str}\n"
        s += '  }'

        return s

    def assume_layers_info(self, shape_in: tuple[int, int, int]) -> None:
        handles = []
        for name, child in self.golden_net.named_children():
            h = child.register_forward_hook(self._layers_info_hook_wrapper(name))
            handles.append(h)

        dummy_input = torch.rand((1, *shape_in, 1)).to(self.device)
        self.golden_net.forward(dummy_input)

        for h in handles:
            h.remove()

    def _layers_info_hook_wrapper(self, layer_name: str) -> Callable[[nn.Module, tuple[Any, ...], Tensor], None]:
        def layers_info_hook(layer: nn.Module, _, output: Tensor) -> None:
            self.layers_info.infer(layer_name, layer, output)
            self.layers[layer_name] = layer

        return layers_info_hook

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
            if f.model.is_parametric() and f.model.param_name not in self.slayer.neuron:
                continue

            is_syn = f.model.is_synaptic()
            to_remove = set()

            for s in f.sites:  # Validate only the defined fault sites
                v = self.layers_info.is_injectable(s.layer)
                if v:
                    shape = self.layers_info.get_shapes(is_syn, s.layer)
                    if is_syn:
                        v &= -shape[0] <= s.position[0] < shape[0]

                    for i in range(1, 4):
                        # shapes_neu index values, si: 0-2
                        # shapes_syn index values, si: 1-3
                        si = i - (not is_syn)
                        v &= -shape[si] <= s.position[i] < shape[si]

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

            for s in f.sites_pending:
                if not s.layer:
                    s.layer = random.choice(self.layers_info.get_injectables())

                shape = self.layers_info.get_shapes(is_syn, s.layer)

                pos = list(s.position)
                if pos[0] is None:
                    pos[0] = random.randrange(shape[0]) if is_syn else slice(None)

                for i in range(1, 4):
                    if pos[i] is None:
                        si = i - (not is_syn)
                        pos[i] = random.randrange(shape[si])

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

    def run(self, test_loader: DataLoader, error: snn.loss = None) -> None:
        self._pre_run()

        self.progress = CampaignProgress(len(test_loader), len(self.rounds))
        self.progress_lock = Lock()

        progr_thread = Thread(target=refresh_progress_job, args=(self.progress, .1,), daemon=True)
        progr_thread.start()

        with torch.no_grad():
            if len(self.rounds) <= 1:
                self._evaluate_single(test_loader, error)
            else:
                self._evaluate_optimized(test_loader, error)

        progr_thread.join()

        self._post_run()

    def _pre_run(self) -> None:
        if not self.rounds:
            self.rounds = [FaultRound()]

        for r, round in enumerate(self.rounds):
            round = self.rounds[r] = FaultRound(sorted(round.items(), key=lambda item: self.layers_info.index(item[0][0])))
            min_lay_name = next(iter(round.keys()), (None,))[0] or 'golden'

            self.round_groups.setdefault(min_lay_name, list())
            self.round_groups[min_lay_name].append(r)

            self._perturb_param(round)

        self.round_groups = dict(sorted(self.round_groups.items(), key=lambda item: -1 if item[0] == 'golden' else self.layers_info.index(item[0])))

    def _post_run(self) -> None:
        for round in self.rounds:
            round.stats.update()
            self._restore_param(round)

    def _evaluate_single(self, test_loader: DataLoader, error: snn.loss = None) -> None:
        round = self.rounds[0]
        is_out_faulty = False

        if round:
            round_max_lay = list(round.keys())[-1][0]
            is_out_faulty = self.layers_info.is_output(round_max_lay)

            if is_out_faulty:
                out_neuron_callable = self._neuron_pre_hook_wrapper(round, self.layers_info.order[-1])
                out_lay = self.layers[self.layers_info.order[-1]]

            handles = self._register_hooks(round)
            self._perturb_synap(round)

        for b, (input, target, label) in enumerate(test_loader):
            for output in self._forward(input.to(self.device)):
                pass

            if is_out_faulty:
                out_neuron_callable(out_lay, (output,))

            self._stats_step(round.stats, output, target, label, error)

            with self.progress_lock:
                self.progress.step()
                self.progress.set_batch(b)

        if round:
            self._restore_synap(round)
            for h in handles:
                h.remove()

    def _evaluate_optimized(self, test_loader: DataLoader, error: snn.loss = None) -> None:
        out_lay_idx = len(self.layers_info.order) - 1
        for b, (input, target, label) in enumerate(test_loader):  # For each batch
            # Store golden spikes
            spikes = [input.to(self.device)] + [spikes for spikes in self._forward(input.to(self.device))]

            for group_lay_name, round_group in self.round_groups.items():  # For each fault round group
                group_lay_idx = self.layers_info.index(group_lay_name)
                spikes_lay_in = spikes[group_lay_idx]

                for r in round_group:  # For each fault round
                    round = self.rounds[r]

                    # TODO: Move out of the loop
                    handles = self._register_hooks(round)
                    self._perturb_synap(round)

                    # TODO: Calculate in _prepare_rounds
                    round_max_lay_name = list(round.keys())[-1][0]
                    round_max_lay_idx = self.layers_info.index(round_max_lay_name)
                    for spikes_round_out in self._forward(spikes_lay_in, group_lay_name, round_max_lay_name):
                        pass

                    # TODO: Use layers info instead of index
                    if round_max_lay_idx != out_lay_idx:  # if not is_out_faulty
                        # Early stop optimization
                        if torch.equal(spikes_round_out, spikes[round_max_lay_idx + 1]):
                            output = spikes[-1]
                        else:
                            for output in self._forward(spikes_round_out, self.layers_info.order[round_max_lay_idx + 1]):
                                pass
                    else:
                        output = spikes_round_out
                        out_neuron_callable = self._neuron_pre_hook_wrapper(round, self.layers_info.order[-1])
                        out_neuron_callable(self.layers[self.layers_info.order[-1]], (output,))

                    # Testing stats
                    self._stats_step(round.stats, output, target, label, error)

                    self._restore_synap(round)
                    for h in handles:
                        h.remove()

                    with self.progress_lock:
                        self.progress.step()

            with self.progress_lock:
                self.progress.set_batch(b)

    def _stats_step(self, stats: snn.utils.stats(), output: Tensor, target: Tensor, label: Tensor, error: snn.loss = None) -> None:
        stats.testing.correctSamples += torch.sum(snn.predict.getClass(output) == label).item()
        stats.testing.numSamples += len(label)
        if error:
            stats.testing.lossSum += error.numSpikes(output, target.to(self.device)).cpu().item()

    def _forward(self, spikes_in: Tensor, layer_start: str = None, layer_end: str = None) -> Iterator[Tensor]:
        s = torch.clone(spikes_in)
        has_started = not layer_start

        for lay_name in self.layers_info.order:
            has_started |= lay_name == layer_start
            if not has_started:
                continue

            # TODO: Make a copy of layer names without the dropout ones to avoid checking inside the loop
            if not isinstance(self.layers[lay_name], snn.slayer._dropoutLayer):
                s = self.slayer.spike(self.slayer.psp(self.layers[lay_name](s)))

            yield s  # Yield next layer's output

            if lay_name == layer_end:
                break

    def _register_hooks(self, round: FaultRound) -> list[RemovableHandle]:
        faults = round.search_neuronal()
        if not faults:
            return []

        handles = []
        for f in faults:
            is_param = f.model.is_parametric()
            for s in f.sites:

                # Neuron fault pre-hooks
                # Neuron faults for last layer are evaluated directly on network's output
                if not self.layers_info.is_output(s.layer):
                    next_lay_name = self.layers_info.get_following(s.layer)
                    h = self.layers[next_lay_name].register_forward_pre_hook(self._neuron_pre_hook_wrapper(round, s.layer))
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
                lay = self.layers[s.layer]
                with torch.no_grad():
                    lay.weight[s.unroll()] = f.model.perturb(lay.weight[s.unroll()], s)

    def _restore_synap(self, round: FaultRound) -> None:
        for f in round.search_synaptic():
            for s in f.sites:
                # Restore weights after the end of the round (no hook needed)
                original = f.model.restore(s)
                if original is not None:
                    with torch.no_grad():
                        self.layers[s.layer].weight[s.unroll()] = original

    def _perturb_param(self, round: FaultRound) -> None:
        # Create a dummy layer for each parametric fault
        for f in round.search_parametric():
            f.model.param_perturb(self.slayer)

    def _restore_param(self, round: FaultRound) -> None:
        # Destroy dummy layers
        for f in round.search_parametric():
            f.model.param_restore()

    def run_complete(self, test_loader: DataLoader, fault_model: FaultModel, layer_names: Iterable[str] = None, error: snn.loss = None) -> None:
        if layer_names is not None and not isinstance(layer_names, Iterable) or isinstance(layer_names, str):
            raise TypeError(f"'{type(layer_names).__name__}' object for layer_names arguement is not iterable or is str")

        self.rounds.clear()
        is_syn = fault_model.is_synaptic()
        lay_names_inj = [lay_name for lay_name in layer_names if self.layers_info.is_injectable(lay_name)] if layer_names else self.layers_info.get_injectables()

        for lay_name in lay_names_inj:
            lay_shape = self.layers_info.get_shapes(is_syn, lay_name)
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
