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

    @property
    def expression(
        self,
    ) -> (
        tuple[ld.amplitudes.Expression, ld.amplitudes.Expression]
        | ld.amplitudes.Expression
    ):
        topology = ld.Topology.missing_k2('beam', ['kshort1', 'kshort2'], 'proton')
        mandelstam = ld.Mandelstam(topology, 't')
        recoil_mass = ld.Mass('proton')
        daughter_1_mass = ld.Mass('kshort1')
        daughter_2_mass = ld.Mass('kshort2')
        resonance_mass = ld.Mass(['kshort1', 'kshort2'])
        angles = ld.Angles(topology, 'kshort1')
        if self.r is None:
            angular_distribution = ld.Ylm(
                f'ylm {self}', self.j.to_int(), self.m, angles
            )
            if self.j == AngularMomentum.S:
                coeff = ld.Scalar(f'coeff {self}', ld.parameter(f'{self} real'))
            else:
                coeff = ld.ComplexScalar(
                    f'coeff {self}',
                    ld.parameter(f'{self} real'),
                    ld.parameter(f'{self} imag'),
                )
            return coeff * angular_distribution
        polarization = ld.Polarization(
            topology, pol_magnitude='pol_magnitude', pol_angle='pol_angle'
        )
        angular_distribution = ld.Zlm(
            f'zlm {self}',
            self.j.to_int(),
            self.m,
            str(self.r.value),
            angles,
            polarization,
        )
        if self.j == AngularMomentum.S:
            coeff = ld.Scalar(f'coeff {self}', ld.parameter(f'{self} real'))
        else:
            coeff = ld.ComplexScalar(
                f'coeff {self}',
                ld.parameter(f'{self} real'),
                ld.parameter(f'{self} imag'),
            )
        kappa = ld.PhaseSpaceFactor(
            f'kappa {self}',
            recoil_mass,
            daughter_1_mass,
            daughter_2_mass,
            resonance_mass,
            mandelstam,
        )
        return (
            coeff * angular_distribution.real() * kappa,
            coeff * angular_distribution.imag() * kappa,
        )

    @property
    def amplitude_names(self) -> list[str]:
        names = [f'coeff {self}', f'{"ylm" if self.r is None else "zlm"} {self}']
        if self.r is not None:
            names.append(f'kappa {self}')
        return names


def build_model(waves: list[Wave]) -> ld.Expression:
    simple_waves = [w for w in waves if w.r is None]
    positive_waves = [w for w in waves if w.positive]
    negative_waves = [w for w in waves if w.negative]
    if simple_waves:
        if positive_waves or negative_waves:
            msg = 'Cannot have both simple and polarized waves'
            raise ValueError(msg)
        terms = [w.expression for w in simple_waves]
        terms = [t for t in terms if isinstance(t, ld.amplitudes.Expression)]
        simple_sum = ld.expr_sum(terms)
        return simple_sum.norm_sqr()
    pos_sum = [w.expression for w in positive_waves]
    pos_sum_re = ld.expr_sum([w[0] for w in pos_sum if isinstance(w, tuple)])
    pos_sum_im = ld.expr_sum([w[1] for w in pos_sum if isinstance(w, tuple)])
    neg_sum = [w.expression for w in negative_waves]
    neg_sum_re = ld.expr_sum([w[0] for w in neg_sum if isinstance(w, tuple)])
    neg_sum_im = ld.expr_sum([w[1] for w in neg_sum if isinstance(w, tuple)])
    return (
        pos_sum_re.norm_sqr()
        + pos_sum_im.norm_sqr()
        + neg_sum_re.norm_sqr()
        + neg_sum_im.norm_sqr()
    )
