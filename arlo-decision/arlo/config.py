"""Configuration for the Arlo decision head."""

from dataclasses import asdict, dataclass


@dataclass
class ArloConfig:
    """Shape of an Arlo head.

    ``hidden_size`` must equal the width of the hidden states of the model Arlo reads. The rest are
    Arlo's own size. With the defaults and ``hidden_size=384`` Arlo has about 0.4 M parameters.
    """

    hidden_size: int
    slots: int = 8              # K latent decision slots
    dim: int = 128              # slot width
    heads: int = 4              # bilinear pointer heads
    pointer_rank: int = 64      # rank of each pointer head
    refine_steps: int = 2       # slot refinement passes
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("hidden_size", "slots", "dim", "heads", "pointer_rank", "refine_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"ArloConfig.{name} must be a positive integer, got {value!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ArloConfig":
        return cls(**data)
