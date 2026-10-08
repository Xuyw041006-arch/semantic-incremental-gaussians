"""Per-batch refinement windows; independent of CUDA and future frames."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RefinementWindow:
    steps: int
    warmup: int = 200
    settle: int = 800
    interval: int = 100

    def __post_init__(self):
        if self.steps < 1 or min(self.warmup, self.settle) < 0 or self.interval < 1:
            raise ValueError('Invalid refinement window')

    @property
    def warmup_end(self):
        # Tiny smoke runs retain a training step, with no forced growth.
        return min(self.warmup, self.steps // 4)

    @property
    def settle_start(self):
        return self.steps - min(self.settle, self.steps // 3)

    def phase(self, local):
        if not 0 <= local < self.steps:
            raise ValueError('Step outside batch')
        if local < self.warmup_end:
            return 'warmup'
        return 'settle' if local >= self.settle_start else 'refine'

    def refine(self, local):
        return self.phase(local) == 'refine' and local > 0 and local % self.interval == 0


def add_stability_arguments(parser):
    parser.add_argument('--stability-mode', choices=['legacy', 'schedule', 'mature'], default='mature',
                        help='legacy: previous behavior; schedule: batch windows; mature: also gate/rate-limit growth')
    parser.add_argument('--batch-warmup', type=int, default=200)
    parser.add_argument('--batch-settle', type=int, default=800)
    parser.add_argument('--growth-fraction', type=float, default=.1)
    parser.add_argument('--min-child-age', type=int, default=200)
    parser.add_argument('--min-fit-observations', type=int, default=12)


def validate_stability(args):
    if min(args.batch_warmup, args.batch_settle, args.min_child_age) < 0:
        raise ValueError('Stability steps must be nonnegative')
    if not 0 < args.growth_fraction <= .5 or args.min_fit_observations < 2:
        raise ValueError('Growth fraction must be in (0, .5]; fit observations at least 2')
