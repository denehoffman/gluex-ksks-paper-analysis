from typing import Callable

import laddu as ld
import polars as pl

from gluex_ksks_paper_analysis.databases import PolarizationData


def add_polarization(data: pl.LazyFrame) -> pl.LazyFrame:
    rcdb_data = PolarizationData()
    return data.with_columns(
        pl.struct('RunNumber', 'beam_e')
        .map_elements(
            lambda s: rcdb_data.get_pol_data(s['RunNumber'], s['beam_e'])
            or {'pol_magnitude': 0.0, 'pol_angle': 0.0},
            return_dtype=pl.Struct(
                {
                    'pol_magnitude': pl.Float64,
                    'pol_angle': pl.Float64,
                }
            ),
        )
        .alias('pol'),
    ).unnest('pol')


def add_m_meson(data: pl.LazyFrame) -> pl.LazyFrame:
    def process(struct) -> float:
        ks1_px = struct['kshort1_px']
        ks1_py = struct['kshort1_py']
        ks1_pz = struct['kshort1_pz']
        ks1_e = struct['kshort1_e']

        ks2_px = struct['kshort2_px']
        ks2_py = struct['kshort2_py']
        ks2_pz = struct['kshort2_pz']
        ks2_e = struct['kshort2_e']

        ks1_lab = ld.Vec4(ks1_px, ks1_py, ks1_pz, ks1_e)
        ks2_lab = ld.Vec4(ks2_px, ks2_py, ks2_pz, ks2_e)
        return (ks1_lab + ks2_lab).m

    return data.with_columns(
        pl.struct(
            'kshort1_px',
            'kshort1_py',
            'kshort1_pz',
            'kshort1_e',
            'kshort2_px',
            'kshort2_py',
            'kshort2_pz',
            'kshort2_e',
        )
        .map_elements(process, return_dtype=pl.Float64)
        .alias('m_meson')
    )


def add_ksb_costheta(data: pl.LazyFrame) -> pl.LazyFrame:
    def process(struct) -> float:
        p_px = struct['proton_px']
        p_py = struct['proton_py']
        p_pz = struct['proton_pz']
        p_e = struct['proton_e']

        ks1_px = struct['kshort1_px']
        ks1_py = struct['kshort1_py']
        ks1_pz = struct['kshort1_pz']
        ks1_e = struct['kshort1_e']

        ks2_px = struct['kshort2_px']
        ks2_py = struct['kshort2_py']
        ks2_pz = struct['kshort2_pz']
        ks2_e = struct['kshort2_e']

        p_lab = ld.Vec4(p_px, p_py, p_pz, p_e)
        ks1_lab = ld.Vec4(ks1_px, ks1_py, ks1_pz, ks1_e)
        ks2_lab = ld.Vec4(ks2_px, ks2_py, ks2_pz, ks2_e)
        com_boost = p_lab + ks1_lab + ks2_lab
        ks1_com = ks1_lab.boost(-com_boost.beta)
        ks2_com = ks2_lab.boost(-com_boost.beta)
        return min(ks1_com.vec3.costheta, ks2_com.vec3.costheta)

    return data.with_columns(
        pl.struct(
            'proton_px',
            'proton_py',
            'proton_pz',
            'proton_e',
            'kshort1_px',
            'kshort1_py',
            'kshort1_pz',
            'kshort1_e',
            'kshort2_px',
            'kshort2_py',
            'kshort2_pz',
            'kshort2_e',
        )
        .map_elements(process, return_dtype=pl.Float64)
        .alias('ksb_costheta')
    )


def add_m_baryon(data: pl.LazyFrame) -> pl.LazyFrame:
    def process(struct) -> float:
        p_px = struct['proton_px']
        p_py = struct['proton_py']
        p_pz = struct['proton_pz']
        p_e = struct['proton_e']

        ks1_px = struct['kshort1_px']
        ks1_py = struct['kshort1_py']
        ks1_pz = struct['kshort1_pz']
        ks1_e = struct['kshort1_e']

        ks2_px = struct['kshort2_px']
        ks2_py = struct['kshort2_py']
        ks2_pz = struct['kshort2_pz']
        ks2_e = struct['kshort2_e']

        p_lab = ld.Vec4(p_px, p_py, p_pz, p_e)
        ks1_lab = ld.Vec4(ks1_px, ks1_py, ks1_pz, ks1_e)
        ks2_lab = ld.Vec4(ks2_px, ks2_py, ks2_pz, ks2_e)
        com_boost = p_lab + ks1_lab + ks2_lab
        ks1_com = ks1_lab.boost(-com_boost.beta)
        ks2_com = ks2_lab.boost(-com_boost.beta)
        ksb_lab = ks1_lab if ks1_com.vec3.costheta < ks2_com.vec3.costheta else ks2_lab
        return (ksb_lab + p_lab).m

    return data.with_columns(
        pl.struct(
            'proton_px',
            'proton_py',
            'proton_pz',
            'proton_e',
            'kshort1_px',
            'kshort1_py',
            'kshort1_pz',
            'kshort1_e',
            'kshort2_px',
            'kshort2_py',
            'kshort2_pz',
            'kshort2_e',
        )
        .map_elements(process, return_dtype=pl.Float64)
        .alias('m_baryon')
    )


def add_hx_angles(data: pl.LazyFrame) -> pl.LazyFrame:
    def process(struct) -> dict[str, float]:
        beam_px = struct['beam_px']
        beam_py = struct['beam_py']
        beam_pz = struct['beam_pz']
        beam_e = struct['beam_e']

        p_px = struct['proton_px']
        p_py = struct['proton_py']
        p_pz = struct['proton_pz']
        p_e = struct['proton_e']

        ks1_px = struct['kshort1_px']
        ks1_py = struct['kshort1_py']
        ks1_pz = struct['kshort1_pz']
        ks1_e = struct['kshort1_e']

        ks2_px = struct['kshort2_px']
        ks2_py = struct['kshort2_py']
        ks2_pz = struct['kshort2_pz']
        ks2_e = struct['kshort2_e']

        beam_lab = ld.Vec4(beam_px, beam_py, beam_pz, beam_e)
        p_lab = ld.Vec4(p_px, p_py, p_pz, p_e)
        ks1_lab = ld.Vec4(ks1_px, ks1_py, ks1_pz, ks1_e)
        ks2_lab = ld.Vec4(ks2_px, ks2_py, ks2_pz, ks2_e)
        com_boost = p_lab + ks1_lab + ks2_lab
        beam_com = beam_lab.boost(-com_boost.beta)
        p_com = p_lab.boost(-com_boost.beta)
        ks1_com = ks1_lab.boost(-com_boost.beta)
        ks2_com = ks2_lab.boost(-com_boost.beta)
        event = ld.Event(
            p4s=[beam_com, p_com, ks1_com, ks2_com],
            aux=[],
            weight=1.0,
            p4_names=['beam', 'proton', 'kshort1', 'kshort2'],
            aliases={'resonance': ['kshort1', 'kshort2']},
        )
        topology = ld.Topology.missing_k2('beam', 'resonance', 'proton')
        angles_hx = ld.Angles(topology, 'kshort1', 'HX')
        return {
            'hx_costheta': angles_hx.costheta.value(event),
            'hx_phi': angles_hx.phi.value(event),
        }

    return data.with_columns(
        pl.struct(
            'beam_px',
            'beam_py',
            'beam_pz',
            'beam_e',
            'proton_px',
            'proton_py',
            'proton_pz',
            'proton_e',
            'kshort1_px',
            'kshort1_py',
            'kshort1_pz',
            'kshort1_e',
            'kshort2_px',
            'kshort2_py',
            'kshort2_pz',
            'kshort2_e',
        )
        .map_elements(
            process,
            return_dtype=pl.Struct(
                {
                    'hx_costheta': pl.Float64,
                    'hx_phi': pl.Float64,
                }
            ),
        )
        .alias('hx_angles')
    ).unnest('hx_angles')


def add_mandelstam_t(data: pl.LazyFrame) -> pl.LazyFrame:
    # NOTE: these add -t and -t', not t and t'
    def process(struct) -> dict[str, float]:
        beam_px = struct['beam_px']
        beam_py = struct['beam_py']
        beam_pz = struct['beam_pz']
        beam_e = struct['beam_e']

        p_px = struct['proton_px']
        p_py = struct['proton_py']
        p_pz = struct['proton_pz']
        p_e = struct['proton_e']

        ks1_px = struct['kshort1_px']
        ks1_py = struct['kshort1_py']
        ks1_pz = struct['kshort1_pz']
        ks1_e = struct['kshort1_e']

        ks2_px = struct['kshort2_px']
        ks2_py = struct['kshort2_py']
        ks2_pz = struct['kshort2_pz']
        ks2_e = struct['kshort2_e']

        beam_lab = ld.Vec4(beam_px, beam_py, beam_pz, beam_e)
        p_lab = ld.Vec4(p_px, p_py, p_pz, p_e)
        ks1_lab = ld.Vec4(ks1_px, ks1_py, ks1_pz, ks1_e)
        ks2_lab = ld.Vec4(ks2_px, ks2_py, ks2_pz, ks2_e)
        event = ld.Event(
            p4s=[beam_lab, p_lab, ks1_lab, ks2_lab],
            aux=[],
            weight=1.0,
            p4_names=['beam', 'proton', 'kshort1', 'kshort2'],
            aliases={'resonance': ['kshort1', 'kshort2']},
        )
        topology = ld.Topology.missing_k2('beam', 'resonance', 'proton')
        mandelstam_t_var = ld.Mandelstam(topology, channel='t')
        resonance_com = topology.k3_com(event)
        beam_com = topology.k1_com(event)
        resonance_com_min = ld.Vec4(
            beam_com.vec3.x / beam_com.vec3.mag * resonance_com.vec3.mag,
            beam_com.vec3.y / beam_com.vec3.mag * resonance_com.vec3.mag,
            beam_com.vec3.z / beam_com.vec3.mag * resonance_com.vec3.mag,
            resonance_com.e,
        )
        mandelstam_t = mandelstam_t_var.value(event)
        mandelstam_t_min = (beam_com - resonance_com_min).mag2
        return {
            'mandelstam_t': -mandelstam_t,
            'reduced_mandelstam_t': -(mandelstam_t - mandelstam_t_min),
        }

    return data.with_columns(
        pl.struct(
            'beam_px',
            'beam_py',
            'beam_pz',
            'beam_e',
            'proton_px',
            'proton_py',
            'proton_pz',
            'proton_e',
            'kshort1_px',
            'kshort1_py',
            'kshort1_pz',
            'kshort1_e',
            'kshort2_px',
            'kshort2_py',
            'kshort2_pz',
            'kshort2_e',
        )
        .map_elements(
            process,
            return_dtype=pl.Struct(
                {
                    'mandelstam_t': pl.Float64,
                    'reduced_mandelstam_t': pl.Float64,
                }
            ),
        )
        .alias('mandelstam')
    ).unnest('mandelstam')


def add_alt_hypos(data: pl.LazyFrame) -> pl.LazyFrame:
    def process(struct) -> dict[str, float]:
        p_px = struct['proton_px']
        p_py = struct['proton_py']
        p_pz = struct['proton_pz']
        p_e = struct['proton_e']

        piplus1_px = struct['piplus1_px']
        piplus1_py = struct['piplus1_py']
        piplus1_pz = struct['piplus1_pz']
        piplus1_e = struct['piplus1_e']

        piminus1_px = struct['piminus1_px']
        piminus1_py = struct['piminus1_py']
        piminus1_pz = struct['piminus1_pz']
        piminus1_e = struct['piminus1_e']

        piplus2_px = struct['piplus2_px']
        piplus2_py = struct['piplus2_py']
        piplus2_pz = struct['piplus2_pz']
        piplus2_e = struct['piplus2_e']

        piminus2_px = struct['piminus2_px']
        piminus2_py = struct['piminus2_py']
        piminus2_pz = struct['piminus2_pz']
        piminus2_e = struct['piminus2_e']

        p_lab = ld.Vec4(p_px, p_py, p_pz, p_e)
        piplus1_lab = ld.Vec4(piplus1_px, piplus1_py, piplus1_pz, piplus1_e)
        piminus1_lab = ld.Vec4(piminus1_px, piminus1_py, piminus1_pz, piminus1_e)
        piplus2_lab = ld.Vec4(piplus2_px, piplus2_py, piplus2_pz, piplus2_e)
        piminus2_lab = ld.Vec4(piminus2_px, piminus2_py, piminus2_pz, piminus2_e)
        return {
            'm_piplus1': piplus1_lab.m,
            'm_piminus1': piminus1_lab.m,
            'm_piplus2': piplus2_lab.m,
            'm_piminu2': piminus2_lab.m,
            'm_piplus1_piminus1': (piplus1_lab + piminus1_lab).m,
            'm_piplus2_piminus2': (piplus2_lab + piminus2_lab).m,
            'm_piplus1_piminus2': (piplus1_lab + piminus2_lab).m,
            'm_piplus2_piminus1': (piplus2_lab + piminus1_lab).m,
            'm_p_piplus1': (p_lab + piplus1_lab).m,
            'm_p_piplus2': (p_lab + piplus2_lab).m,
            'm_p_piminus1': (p_lab + piminus1_lab).m,
            'm_p_piminus2': (p_lab + piminus2_lab).m,
            'm_p_piplus1_piminus1': (p_lab + piplus1_lab + piminus1_lab).m,
            'm_p_piplus1_piminus2': (p_lab + piplus1_lab + piminus2_lab).m,
            'm_p_piplus2_piminus1': (p_lab + piplus2_lab + piminus1_lab).m,
            'm_p_piplus2_piminus2': (p_lab + piplus2_lab + piminus2_lab).m,
            'm_p_piminus1_piminus2': (p_lab + piminus1_lab + piminus2_lab).m,
        }

    return data.with_columns(
        pl.struct(
            'proton_px',
            'proton_py',
            'proton_pz',
            'proton_e',
            'piplus1_px',
            'piplus1_py',
            'piplus1_pz',
            'piplus1_e',
            'piminus1_px',
            'piminus1_py',
            'piminus1_pz',
            'piminus1_e',
            'piplus2_px',
            'piplus2_py',
            'piplus2_pz',
            'piplus2_e',
            'piminus2_px',
            'piminus2_py',
            'piminus2_pz',
            'piminus2_e',
        )
        .map_elements(
            process,
            return_dtype=pl.Struct(
                {
                    'm_piplus1': pl.Float64,
                    'm_piminus1': pl.Float64,
                    'm_piplus2': pl.Float64,
                    'm_piminu2': pl.Float64,
                    'm_piplus1_piminus1': pl.Float64,
                    'm_piplus2_piminus2': pl.Float64,
                    'm_piplus1_piminus2': pl.Float64,
                    'm_piplus2_piminus1': pl.Float64,
                    'm_p_piplus1': pl.Float64,
                    'm_p_piplus2': pl.Float64,
                    'm_p_piminus1': pl.Float64,
                    'm_p_piminus2': pl.Float64,
                    'm_p_piplus1_piminus1': pl.Float64,
                    'm_p_piplus1_piminus2': pl.Float64,
                    'm_p_piplus2_piminus1': pl.Float64,
                    'm_p_piplus2_piminus2': pl.Float64,
                    'm_p_piminus1_piminus2': pl.Float64,
                }
            ),
        )
        .alias('m_alt_hypos')
    ).unnest('m_alt_hypos')


VARIABLE_MAPPINGS: dict[str, Callable[[pl.LazyFrame], pl.LazyFrame]] = {
    'pol_magnitude': add_polarization,
    'pol_angle': add_polarization,
    'polarization': add_polarization,
    'm_meson': add_m_meson,
    'ksb_costheta': add_ksb_costheta,
    'm_baryon': add_m_baryon,
    'hx_angles': add_hx_angles,
    'hx_costheta': add_hx_angles,
    'hx_phi': add_hx_angles,
    'mandelstam_t': add_mandelstam_t,
    'reduced_mandelstam_t': add_mandelstam_t,
    'alt_hypos': add_alt_hypos,
    'm_piplus1': add_alt_hypos,
    'm_piminus1': add_alt_hypos,
    'm_piplus2': add_alt_hypos,
    'm_piminus2': add_alt_hypos,
    'm_piplus1_piminus1': add_alt_hypos,
    'm_piplus2_piminus2': add_alt_hypos,
    'm_piplus1_piminus2': add_alt_hypos,
    'm_piplus2_piminus1': add_alt_hypos,
    'm_p_piplus1': add_alt_hypos,
    'm_p_piplus2': add_alt_hypos,
    'm_p_piminus1': add_alt_hypos,
    'm_p_piminus2': add_alt_hypos,
    'm_p_piplus1_piminus1': add_alt_hypos,
    'm_p_piplus1_piminus2': add_alt_hypos,
    'm_p_piplus2_piminus1': add_alt_hypos,
    'm_p_piplus2_piminus2': add_alt_hypos,
    'm_p_piminus1_piminus2': add_alt_hypos,
}


def add_variable(name: str, df: pl.LazyFrame) -> pl.LazyFrame:
    if name in df:
        return df
    mapping = VARIABLE_MAPPINGS.get(name)
    if mapping is not None:
        return mapping(df)
    return df
