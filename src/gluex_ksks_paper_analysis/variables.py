from typing import Callable

import laddu as ld
import polars as pl

from gluex_ksks_paper_analysis.databases import RCDBData


def add_polarization(data: pl.LazyFrame) -> pl.LazyFrame:
    rcdb_data = RCDBData()
    return (
        data.with_columns(
            pl.struct('RunNumber', 'p4_0_E')
            .map_elements(
                lambda s: rcdb_data.get_pol_data(s['RunNumber'], s['p4_0_E'])
                or {'x': 0.0, 'y': 0.0, 'is_polarized': 0.0},
                return_dtype=pl.Struct(
                    {'x': pl.Float64, 'y': pl.Float64, 'is_polarized': pl.Float64}
                ),
            )
            .alias('pol'),
        )
        .unnest('pol')
        .with_columns(
            aux_0_x=pl.col('x').cast(pl.Float32),
            aux_0_y=pl.col('y').cast(pl.Float32),
            aux_0_z=pl.lit(0.0, dtype=pl.Float32),
        )
        .drop('x', 'y')
    )


def add_m_meson(data: pl.LazyFrame) -> pl.LazyFrame:
    def process(struct) -> float:
        ks1_px = struct['p4_2_Px']
        ks1_py = struct['p4_2_Py']
        ks1_pz = struct['p4_2_Pz']
        ks1_e = struct['p4_2_E']

        ks2_px = struct['p4_3_Px']
        ks2_py = struct['p4_3_Py']
        ks2_pz = struct['p4_3_Pz']
        ks2_e = struct['p4_3_E']

        ks1_lab = ld.Vec4(ks1_px, ks1_py, ks1_pz, ks1_e)
        ks2_lab = ld.Vec4(ks2_px, ks2_py, ks2_pz, ks2_e)
        return (ks1_lab + ks2_lab).m

    return data.with_columns(
        pl.struct(
            'p4_2_Px',
            'p4_2_Py',
            'p4_2_Pz',
            'p4_2_E',
            'p4_3_Px',
            'p4_3_Py',
            'p4_3_Pz',
            'p4_3_E',
        )
        .map_elements(process, return_dtype=pl.Float64)
        .alias('m_meson')
    )


def add_ksb_costheta(data: pl.LazyFrame) -> pl.LazyFrame:
    def process(struct) -> float:
        p_px = struct['p4_1_Px']
        p_py = struct['p4_1_Py']
        p_pz = struct['p4_1_Pz']
        p_e = struct['p4_1_E']

        ks1_px = struct['p4_2_Px']
        ks1_py = struct['p4_2_Py']
        ks1_pz = struct['p4_2_Pz']
        ks1_e = struct['p4_2_E']

        ks2_px = struct['p4_3_Px']
        ks2_py = struct['p4_3_Py']
        ks2_pz = struct['p4_3_Pz']
        ks2_e = struct['p4_3_E']

        p_lab = ld.Vec4(p_px, p_py, p_pz, p_e)
        ks1_lab = ld.Vec4(ks1_px, ks1_py, ks1_pz, ks1_e)
        ks2_lab = ld.Vec4(ks2_px, ks2_py, ks2_pz, ks2_e)
        com_boost = p_lab + ks1_lab + ks2_lab
        ks1_com = ks1_lab.boost(-com_boost.beta)
        ks2_com = ks2_lab.boost(-com_boost.beta)
        return min(ks1_com.vec3.costheta, ks2_com.vec3.costheta)

    return data.with_columns(
        pl.struct(
            'p4_1_Px',
            'p4_1_Py',
            'p4_1_Pz',
            'p4_1_E',
            'p4_2_Px',
            'p4_2_Py',
            'p4_2_Pz',
            'p4_2_E',
            'p4_3_Px',
            'p4_3_Py',
            'p4_3_Pz',
            'p4_3_E',
        )
        .map_elements(process, return_dtype=pl.Float64)
        .alias('ksb_costheta')
    )


def add_m_baryon(data: pl.LazyFrame) -> pl.LazyFrame:
    def process(struct) -> float:
        p_px = struct['p4_1_Px']
        p_py = struct['p4_1_Py']
        p_pz = struct['p4_1_Pz']
        p_e = struct['p4_1_E']

        ks1_px = struct['p4_2_Px']
        ks1_py = struct['p4_2_Py']
        ks1_pz = struct['p4_2_Pz']
        ks1_e = struct['p4_2_E']

        ks2_px = struct['p4_3_Px']
        ks2_py = struct['p4_3_Py']
        ks2_pz = struct['p4_3_Pz']
        ks2_e = struct['p4_3_E']

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
            'p4_1_Px',
            'p4_1_Py',
            'p4_1_Pz',
            'p4_1_E',
            'p4_2_Px',
            'p4_2_Py',
            'p4_2_Pz',
            'p4_2_E',
            'p4_3_Px',
            'p4_3_Py',
            'p4_3_Pz',
            'p4_3_E',
        )
        .map_elements(process, return_dtype=pl.Float64)
        .alias('m_baryon')
    )


def add_hx_angles(data: pl.LazyFrame) -> pl.LazyFrame:
    def process(struct) -> dict[str, float]:
        beam_px = struct['p4_0_Px']
        beam_py = struct['p4_0_Py']
        beam_pz = struct['p4_0_Pz']
        beam_e = struct['p4_0_E']

        p_px = struct['p4_1_Px']
        p_py = struct['p4_1_Py']
        p_pz = struct['p4_1_Pz']
        p_e = struct['p4_1_E']

        ks1_px = struct['p4_2_Px']
        ks1_py = struct['p4_2_Py']
        ks1_pz = struct['p4_2_Pz']
        ks1_e = struct['p4_2_E']

        ks2_px = struct['p4_3_Px']
        ks2_py = struct['p4_3_Py']
        ks2_pz = struct['p4_3_Pz']
        ks2_e = struct['p4_3_E']

        beam_lab = ld.Vec4(beam_px, beam_py, beam_pz, beam_e)
        p_lab = ld.Vec4(p_px, p_py, p_pz, p_e)
        ks1_lab = ld.Vec4(ks1_px, ks1_py, ks1_pz, ks1_e)
        ks2_lab = ld.Vec4(ks2_px, ks2_py, ks2_pz, ks2_e)
        com_boost = p_lab + ks1_lab + ks2_lab
        beam_com = beam_lab.boost(-com_boost.beta)
        p_com = p_lab.boost(-com_boost.beta)
        ks1_com = ks1_lab.boost(-com_boost.beta)
        ks2_com = ks2_lab.boost(-com_boost.beta)
        event = ld.Event(p4s=[beam_com, p_com, ks1_com, ks2_com], aux=[], weight=1.0)
        angles_hx = ld.Angles(0, [1], [2], [2, 3], 'HX')
        return {
            'hx_costheta': angles_hx.costheta.value(event),
            'hx_phi': angles_hx.phi.value(event),
        }

    return data.with_columns(
        pl.struct(
            'p4_0_Px',
            'p4_0_Py',
            'p4_0_Pz',
            'p4_0_E',
            'p4_1_Px',
            'p4_1_Py',
            'p4_1_Pz',
            'p4_1_E',
            'p4_2_Px',
            'p4_2_Py',
            'p4_2_Pz',
            'p4_2_E',
            'p4_3_Px',
            'p4_3_Py',
            'p4_3_Pz',
            'p4_3_E',
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
        beam_px = struct['p4_0_Px']
        beam_py = struct['p4_0_Py']
        beam_pz = struct['p4_0_Pz']
        beam_e = struct['p4_0_E']

        p_px = struct['p4_1_Px']
        p_py = struct['p4_1_Py']
        p_pz = struct['p4_1_Pz']
        p_e = struct['p4_1_E']

        ks1_px = struct['p4_2_Px']
        ks1_py = struct['p4_2_Py']
        ks1_pz = struct['p4_2_Pz']
        ks1_e = struct['p4_2_E']

        ks2_px = struct['p4_3_Px']
        ks2_py = struct['p4_3_Py']
        ks2_pz = struct['p4_3_Pz']
        ks2_e = struct['p4_3_E']

        beam_lab = ld.Vec4(beam_px, beam_py, beam_pz, beam_e)
        p_lab = ld.Vec4(p_px, p_py, p_pz, p_e)
        ks1_lab = ld.Vec4(ks1_px, ks1_py, ks1_pz, ks1_e)
        ks2_lab = ld.Vec4(ks2_px, ks2_py, ks2_pz, ks2_e)
        event = ld.Event(
            p4s=[beam_lab, p_lab, ks1_lab, ks2_lab], aux=[], weight=1.0
        ).boost_to_rest_frame_of([1, 2, 3])
        mandelstam_t_var = ld.Mandelstam([0], [], [2, 3], [1], channel='t')
        resonance_com = event.get_p4_sum([2, 3])
        beam_com = event.p4s[0]
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
            'p4_0_Px',
            'p4_0_Py',
            'p4_0_Pz',
            'p4_0_E',
            'p4_1_Px',
            'p4_1_Py',
            'p4_1_Pz',
            'p4_1_E',
            'p4_2_Px',
            'p4_2_Py',
            'p4_2_Pz',
            'p4_2_E',
            'p4_3_Px',
            'p4_3_Py',
            'p4_3_Pz',
            'p4_3_E',
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
        p_px = struct['p4_1_Px']
        p_py = struct['p4_1_Py']
        p_pz = struct['p4_1_Pz']
        p_e = struct['p4_1_E']

        piplus1_px = struct['p4_4_Px']
        piplus1_py = struct['p4_4_Py']
        piplus1_pz = struct['p4_4_Pz']
        piplus1_e = struct['p4_4_E']

        piminus1_px = struct['p4_5_Px']
        piminus1_py = struct['p4_5_Py']
        piminus1_pz = struct['p4_5_Pz']
        piminus1_e = struct['p4_5_E']

        piplus2_px = struct['p4_6_Px']
        piplus2_py = struct['p4_6_Py']
        piplus2_pz = struct['p4_6_Pz']
        piplus2_e = struct['p4_6_E']

        piminus2_px = struct['p4_7_Px']
        piminus2_py = struct['p4_7_Py']
        piminus2_pz = struct['p4_7_Pz']
        piminus2_e = struct['p4_7_E']

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
            'p4_1_Px',
            'p4_1_Py',
            'p4_1_Pz',
            'p4_1_E',
            'p4_4_Px',
            'p4_4_Py',
            'p4_4_Pz',
            'p4_4_E',
            'p4_5_Px',
            'p4_5_Py',
            'p4_5_Pz',
            'p4_5_E',
            'p4_6_Px',
            'p4_6_Py',
            'p4_6_Pz',
            'p4_6_E',
            'p4_7_Px',
            'p4_7_Py',
            'p4_7_Pz',
            'p4_7_E',
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
    'aux_0_x': add_polarization,
    'aux_0_y': add_polarization,
    'aux_0_z': add_polarization,
    'is_polarized': add_polarization,
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
