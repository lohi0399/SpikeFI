from collections.abc import Callable, Iterable
from copy import deepcopy
import random
from threading import Lock, Thread
from types import MethodType
from typing import Any, Optional

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

import slayerSNN as snn
from slayerSNN.slayer import spikeLayer

from .fault import Fault, FaultModel, FaultRound, FaultSite
from .utils.layer import LayersInfo
from .utils.progress import CampaignProgress, refresh_progress_job


# FIXME: Fix long lines
# TODO: Logging + silent
# TODO: Results manipulation (read/write results in io.py and save/load results here)


class Campaign:
    def __init__(self, net: nn.Module, shape_in: tuple[int, int, int], slayer: spikeLayer) -> None:
        self.golden = net
        self.slayer = slayer
        self.faulty: list[nn.Module] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.layers_info = LayersInfo()
        self.infer_layers_info(shape_in)

        self.rounds = [FaultRound()]
        self.round_groups: dict[str, list[int]] = {}  # TODO: Remove?

    def __repr__(self) -> str:
        s = 'FI Campaign:\n'
        s += f"  - Network: '{self.golden.__class__.__name__}':\n"
        s += f"  - {str(self.layers_info).replace('}', '  }')}\n"
        s += f"  - Rounds ({len(self.rounds)}): {{\n"
        for round_idx, round in enumerate(self.rounds):
            round_str = str(round).replace('\n', '\n      ')
            s += f"      #{round_idx}: {round_str}\n"
        s += '  }'

        return s

    def infer_layers_info(self, shape_in: tuple[int, int, int]) -> None:
        handles = []
        for name, child in self.golden.named_children():
            hook = self.layers_info.infer_hook_wrapper(name)
            handle = child.register_forward_hook(hook)
            handles.append(handle)

        dummy_input = torch.rand((1, *shape_in, 1)).to(self.device)
        self.golden(dummy_input)

        for handle in handles:
            handle.remove()

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
            # Sort fault round's faults in ascending order of faults appearence (early faulty layer first)
            round = self.rounds[r] = FaultRound(sorted(round.items(), key=lambda item: self.layers_info.index(item[0][0])))
            early_lay_name = next(iter(round.keys()), (None,))[0] or 'golden'

            # Group fault rounds per earliest faulty layer
            self.round_groups.setdefault(early_lay_name, list())
            self.round_groups[early_lay_name].append(r)

            # Create fault round's faulty network instance
            self.faulty.append(self._perturb_net(round))

        # Sort fault round goups in ascending order of groups earliest layer
        self.round_groups = dict(sorted(self.round_groups.items(), key=lambda item: -1 if item[0] == 'golden' else self.layers_info.index(item[0])))

        # Assign optimized forward function to golden network
        self.golden.forward_opt = MethodType(Campaign._forward_opt_wrapper(self.layers_info, self.slayer), self.golden)

    def _post_run(self) -> None:
        # Update round statistics
        for round in self.rounds:
            round.stats.update()

        # Destroy variables needed in run method
        self.faulty.clear()
        self.round_groups.clear()

    def _perturb_net(self, sorted_round: FaultRound) -> nn.Module:
        faulty = deepcopy(self.golden)

        for layer in self.layers_info.get_injectables():
            neuronal = sorted_round.search_neuronal(layer)
            parametric = sorted_round.search_parametric(layer)
            synaptic = sorted_round.search_synaptic(layer)

            # Neuronal faults
            if neuronal:
                # Neuronal faults for last layer are evaluated directly on faulty network's output
                following_layer = self.layers_info.get_following(layer)
                if not following_layer:
                    continue

                # Register neuron fault pre-hooks
                pre_hook = Campaign._neuron_pre_hook_wrapper(neuronal)
                layer = getattr(faulty, following_layer)
                layer.register_forward_pre_hook(pre_hook)

                # Parametric faults (subset of neuronal faults)
                if parametric:
                    # Register parametric fault hooks
                    hook = Campaign._parametric_hook_wrapper(parametric)
                    layer = getattr(faulty, layer)
                    layer.register_forward_hook(hook)

                    for fault in parametric:
                        # Create parametric faults' dummy layers
                        fault.model.param_perturb(self.slayer)

            # Synaptic faults
            for fault in synaptic:
                for site in fault.sites:
                    # Perturb weights for synapse faults
                    layer = getattr(faulty, layer)
                    with torch.no_grad():
                        layer.weight[site.unroll()] = fault.model.perturb(layer.weight[site.unroll()], site)

        faulty.forward_opt = MethodType(Campaign._forward_opt_wrapper(self.layers_info, self.slayer), faulty)

        return faulty

    @staticmethod
    def _forward_opt_wrapper(layers_info: LayersInfo, slayer: spikeLayer) -> Callable[[Tensor, Optional[str], Optional[str]], Tensor]:
        def forward_opt(self: nn.Module, spikes_in: Tensor, start_layer: str = None, end_layer: str = None) -> Tensor:
            start_layer_idx = layers_info.index(start_layer) if start_layer else 0
            end_layer_idx = layers_info.index(end_layer) if end_layer else len(layers_info) - 1

            subject_layers = [lay for lay in layers_info.order
                              if start_layer_idx <= layers_info.index(lay) <= end_layer_idx]

            spikes = spikes_in
            for layer_name in subject_layers:
                layer = getattr(self, layer_name)
                spikes = layer(spikes)
                if not isinstance(layer, snn.slayer._dropoutLayer):
                    # Dropout layers are not useful in inference but they may have registered (pre-)hooks
                    spikes = slayer.spike(slayer.psp(spikes))

            return spikes

        return forward_opt

    def _evaluate_single(self, test_loader: DataLoader, error: snn.loss = None) -> None:
        round = self.rounds[0]
        is_out_faulty = False

        if round:
            late_layer = list(round.keys())[-1][0]
            is_out_faulty = self.layers_info.is_output(late_layer)

            if is_out_faulty:
                out_layer_name = self.layers_info.order[-1]
                out_layer = getattr(self.faulty[0], out_layer_name)
                out_neuron_callable = self._neuron_pre_hook_wrapper(round.search_neuronal(out_layer_name))

        for b, (input, target, label) in enumerate(test_loader):
            output = self.faulty[0].forward(input.to(self.device))
            if is_out_faulty:
                out_neuron_callable(out_layer, (output,))

            self._stats_step(round.stats, output, target, label, error)

            with self.progress_lock:
                self.progress.step()
                self.progress.set_batch(b)

    def _evaluate_optimized(self, test_loader: DataLoader, error: snn.loss = None) -> None:
        out_lay_idx = len(self.layers_info.order) - 1
        out_layer_name = self.layers_info.order[-1]
        for b, (input, target, label) in enumerate(test_loader):  # For each batch
            # Store golden spikes
            golden_spikes = [input.to(self.device)]
            for layer_idx, layer_name in enumerate(self.layers_info.order):
                golden_spikes.append(self.golden.forward_opt(golden_spikes[layer_idx], layer_name, layer_name))

            # TODO: Check if round has no faults and return nominal spikes directly
            for group_lay_name, round_group in self.round_groups.items():  # For each fault round group
                group_lay_idx = self.layers_info.index(group_lay_name)
                spikes_lay_in = golden_spikes[group_lay_idx]

                for r in round_group:  # For each fault round
                    round = self.rounds[r]
                    faulty = self.faulty[r]

                    # TODO: Calculate once and store
                    round_late_lay_name = list(round.keys())[-1][0]
                    round_late_lay_idx = self.layers_info.index(round_late_lay_name)

                    spikes_round_out = faulty.forward_opt(spikes_lay_in, group_lay_name, round_late_lay_name)

                    # FIXME: Early stop optimization needs to compare with layer's output that follows the late layer
                    if round_late_lay_idx != out_lay_idx:  # if not is_out_faulty
                        # Early stop optimization
                        early_stop = torch.equal(spikes_round_out, golden_spikes[round_late_lay_idx + 1])
                        output = golden_spikes[-1] if early_stop else \
                            faulty.forward(spikes_round_out, self.layers_info.get_following(round_late_lay_name))
                    else:
                        output = spikes_round_out
                        # TODO: Calculate once and store
                        out_neuron_callable = self._neuron_pre_hook_wrapper(round.search_neuronal(out_layer_name))
                        out_layer = getattr(faulty, out_layer_name)
                        out_neuron_callable(out_layer, (output,))

                    # Testing stats
                    self._stats_step(round.stats, output, target, label, error)

                    with self.progress_lock:
                        self.progress.step()

            with self.progress_lock:
                self.progress.set_batch(b)

    def _stats_step(self, stats: snn.utils.stats(), output: Tensor, target: Tensor, label: Tensor, error: snn.loss = None) -> None:
        stats.testing.correctSamples += torch.sum(snn.predict.getClass(output) == label).item()
        stats.testing.numSamples += len(label)
        if error:
            stats.testing.lossSum += error.numSpikes(output, target.to(self.device)).cpu().item()

    # TODO: Accept Iterable[FaultModel]
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

    @staticmethod
    def _neuron_pre_hook_wrapper(layer_neuron_faults: list[Fault]) -> Callable[[nn.Module, tuple[Any, ...]], None]:
        def neuron_pre_hook(_, args: tuple[Any, ...]) -> None:
            prev_spikes_out = args[0]
            for fault in layer_neuron_faults:
                for site in fault.sites:
                    # prev_spikes_out[site.unroll()] = fault.model.perturb(prev_spikes_out[site.unroll()], site)
                    prev_spikes_out[site.unroll()] = -1

        return neuron_pre_hook

    @staticmethod
    def _parametric_hook_wrapper(layer_param_faults: list[Fault]) -> Callable[[nn.Module, tuple[Any, ...]], None]:
        def parametric_hook(_, __, spikes_out: Tensor) -> None:
            for fault in layer_param_faults:
                flayer = fault.model.flayer
                fspike_out = flayer.spike(flayer.psp(spikes_out))
                for site in fault.sites:
                    fault.model.args[0][site] = fspike_out[site.unroll()]

        return parametric_hook
