from torch import nn, Tensor

from slayerSNN import slayer


class LayersInfo:
    injectable_types = (slayer._convLayer, slayer._denseLayer)
    unsupported_types = (slayer._pspLayer, slayer._pspFilter, slayer._delayLayer)

    def __init__(self) -> None:
        self.names: set[str] = set()
        self.order: list[str] = []
        self.types: dict[str, str] = {}
        self.injectables: dict[str, bool] = {}
        self.weightables: dict[str, bool] = {}
        self.shapes_neu: dict[str, tuple[int, int, int]] = {}
        self.shapes_syn: dict[str, tuple[int, int, int, int]] = {}

    def __len__(self) -> int:
        return len(self.order)

    def __repr__(self) -> str:
        s = f"Layers info ({len(self.names)} in {len(self.order)} levels): {{\n"
        for lay_idx, lay_name in enumerate(self.order):
            neu = "{:2d} x {:2d} x {:2d}".format(*self.shapes_neu[lay_name])
            syn = "{:2d} x {:2d} x {:2d} x {:2d}".format(*self.shapes_syn[lay_name]) if self.weightables[lay_name] else "-"

            s += f"      #{lay_idx:2d}: '{lay_name}' - {self.types[lay_name]} - {'' if self.injectables[lay_name] else 'non '}injectable\n"
            s += f"        Shapes: neurons {neu} | synapses {syn}\n"
        s += '}'

        return s

    def infer(self, name: str, layer: nn.Module, output: Tensor) -> None:
        if not LayersInfo.is_module_supported(layer):
            print('Attention: Unsupported layer type ' + type(layer) + ' found. Potential invalidity of results.')
            return

        is_injectable = LayersInfo.is_module_injectable(layer)
        if is_injectable and name in self.names:
            print('Cannot use an injectable layer more than once in the network.')
            return

        has_weigths = isinstance(layer, nn.Conv3d) or isinstance(layer, nn.ConvTranspose3d)

        self.names.add(name)
        self.order.append(name)
        self.types[name] = type(layer).__name__
        self.injectables[name] = is_injectable
        self.weightables[name] = has_weigths
        self.shapes_neu[name] = tuple(output.shape[1:4])
        self.shapes_syn[name] = tuple(layer.weight.shape[0:4]) if has_weigths else None

    def index(self, injectable_name: str) -> int:
        is_inj = self.is_injectable(injectable_name)
        if is_inj is None:
            raise ValueError(f"No layer with name '{injectable_name}' found.")
        if not is_inj:
            raise ValueError(f"Layer '{injectable_name}' is not injectable. Only injectable layers support indexing.")

        return self.order.index(injectable_name)

    def get_injectables(self) -> list[str]:
        return [inj[0] for inj in self.injectables.items() if inj[1]]

    def get_following(self, injectable_name: str) -> str | None:
        idx = self.index(injectable_name)
        if idx < len(self) - 1:
            return self.order[idx + 1]

        return None

    def get_shapes(self, syn_select: bool, name: str) -> tuple[int, ...]:
        return self.shapes_syn[name] if syn_select else self.shapes_neu[name]

    def is_injectable(self, name: str) -> bool:
        return self.injectables.get(name, False)

    def is_weightable(self, name: str) -> bool:
        return self.weightables.get(name, False)

    def is_output(self, name: str) -> bool:
        return name == self.order[-1]

    @staticmethod
    def is_module_injectable(layer: nn.Module) -> bool:
        is_inj = False
        for type_ in LayersInfo.injectable_types:
            is_inj |= isinstance(layer, type_)

        return is_inj

    @staticmethod
    def is_module_supported(layer: nn.Module) -> bool:
        is_sup = True
        for type_ in LayersInfo.unsupported_types:
            is_sup &= not isinstance(layer, type_)

        return is_sup
