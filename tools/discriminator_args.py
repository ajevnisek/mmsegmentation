from dataclasses import dataclass


@dataclass
class DiscriminatorArgs:
    """Class for holding discriminator arguments."""
    gan_mode: str = 'wgangp'
    learning_rate: float = 0.0002
    beta1: float = 0.5
    gp_ratio: float = 1.0
