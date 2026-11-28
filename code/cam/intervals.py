"""Single-bias enclosure and contained refinement, CAM equations (3.11)–(3.14)."""
from dataclasses import dataclass
from math import isfinite, sqrt


@dataclass(frozen=True)
class Enclosure:
    bias: float
    T: float
    L: float
    dual: float
    step: float
    center: float
    radius: float

    @property
    def endpoints(self):
        return self.center - self.radius, self.center + self.radius

    def refine(self, energy_at):
        """Reuse the central dual; keep the better feasible energy on each side."""
        k, b = self.step, self.bias
        if k == 0:
            return self.endpoints
        E = self.T/2 + self.L
        correction = self.T*(k/b)**2/2
        left = 0. if k == b else energy_at(b-k)
        right = energy_at(b+k)
        if not all(map(isfinite, (left, right))):
            raise ArithmeticError("Non-finite neighboring energy")
        upper_left = min(left, E - k*self.center + correction)
        upper_right = min(right, E + k*self.center + correction)
        lo, hi = (self.dual-upper_left)/k, (upper_right-self.dual)/k
        initial_lo, initial_hi = self.endpoints
        if hi < lo or lo < initial_lo-2e-9 or hi > initial_hi+2e-9:
            raise ArithmeticError("Nested-interval check failed")
        return lo, hi


def single_bias(b, T, L, dual, *, certified_zero_gap=False):
    if not all(map(isfinite,(b,T,L,dual))) or b <= 0 or T <= 0:
        raise ValueError("Positive finite bias and gradient norm required")
    gap = T/2 + L - dual
    center = (T+L)/b
    if gap == 0 and certified_zero_gap:
        return Enclosure(b,T,L,dual,0.,center,0.)
    if gap <= 0:
        raise ArithmeticError("Nonpositive numerical gap: re-evaluate the energies")
    k = b*min(1.,sqrt(2*gap/T))
    radius = gap/k + T*k/(2*b*b)
    return Enclosure(b,T,L,dual,k,center,radius)

