from __future__ import annotations

from enum import Enum

import laddu as ld


class Reflectivity(Enum):
    Positive = '+'
    Negative = '-'


class AngularMomentum(Enum):
    S = 'S'
    D = 'D'

    def to_int(self) -> int:
        if self == AngularMomentum.S:
            return 0
        return 2


class Wave:
    def __init__(self, s: str):
        self.j = AngularMomentum(s[0])
        self.m = int(s[1:3])
        self.r = Reflectivity(s[3]) if len(s) > 3 else None  # noqa: PLR2004

    j: AngularMomentum
    m: int
    r: Reflectivity | None

    def __str__(self) -> str:
        return f'{self.j.value}{self.m:+}{self.r.value if self.r else ""}'

    def __repr__(self) -> str:
        return f'Wave(j={self.j.value}, m={self.m}{f", r={self.r.value}" if self.r else ""})'

    @property
    def latex(self) -> str:
        if self.r is None:
            return rf'${self.j.value}_{{{self.m}}}$'
        return rf'${self.j.value}_{{{self.m}}}^{{({self.r.value})}}$'

    @property
    def positive(self) -> bool:
        if self.r is None:
            return False
        return self.r == Reflectivity.Positive

    @property
    def negative(self) -> bool:
        if self.r is None:
            return False
        return self.r == Reflectivity.Negative

    def register(
        self, m: ld.Manager
    ) -> (
        tuple[ld.amplitudes.Expression, ld.amplitudes.Expression]
        | ld.amplitudes.Expression
    ):
        angles = ld.Angles(0, [1], [2], [2, 3])
        if self.r is None:
            angular_distribution = m.register(
                ld.Ylm(f'ylm {self}', self.j.to_int(), self.m, angles)
            )
            if self.j == AngularMomentum.S:
                coeff = m.register(
                    ld.Scalar(f'coeff {self}', ld.parameter(f'{self} real'))
                )
            else:
                coeff = m.register(
                    ld.ComplexScalar(
                        f'coeff {self}',
                        ld.parameter(f'{self} real'),
                        ld.parameter(f'{self} imag'),
                    )
                )
            return coeff * angular_distribution
        polarization = ld.Polarization(0, [1], 0)
        angular_distribution = m.register(
            ld.Zlm(
                f'zlm {self}',
                self.j.to_int(),
                self.m,
                str(self.r.value),
                angles,
                polarization,
            )
        )
        if self.j == AngularMomentum.S:
            coeff = m.register(ld.Scalar(f'coeff {self}', ld.parameter(f'{self} real')))
        else:
            coeff = m.register(
                ld.ComplexScalar(
                    f'coeff {self}',
                    ld.parameter(f'{self} real'),
                    ld.parameter(f'{self} imag'),
                )
            )
        return (
            coeff * angular_distribution.real(),
            coeff * angular_distribution.imag(),
        )

    @property
    def amplitude_names(self) -> list[str]:
        return [f'coeff {self}', f'{"ylm" if self.r is None else "zlm"} {self}']


def build_model(waves: list[Wave]) -> ld.Model:
    m = ld.Manager()
    simple_waves = [w for w in waves if w.r is None]
    positive_waves = [w for w in waves if w.positive]
    negative_waves = [w for w in waves if w.negative]
    if simple_waves:
        if positive_waves or negative_waves:
            msg = 'Cannot have both simple and polarized waves'
            raise ValueError(msg)
        simple_sum = ld.amplitude_sum([w.register(m) for w in simple_waves])
        expr = simple_sum.norm_sqr()
        print(expr)
        return m.model(expr)
    pos_sum = [w.register(m) for w in positive_waves]
    pos_sum_re = ld.amplitude_sum([w[0] for w in pos_sum])
    pos_sum_im = ld.amplitude_sum([w[1] for w in pos_sum])
    neg_sum = [w.register(m) for w in negative_waves]
    neg_sum_re = ld.amplitude_sum([w[0] for w in neg_sum])
    neg_sum_im = ld.amplitude_sum([w[1] for w in neg_sum])
    expr = (
        pos_sum_re.norm_sqr()
        + pos_sum_im.norm_sqr()
        + neg_sum_re.norm_sqr()
        + neg_sum_im.norm_sqr()
    )
    print(expr)
    return m.model(expr)
