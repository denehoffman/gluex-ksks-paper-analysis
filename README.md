# Data Analysis for the KsKs Channel at GlueX

This repository contains the data analysis for the $`\gamma p \to K_S^0 K_S^0 p`$ channel at GlueX.

## Usage

1. Clone the repository:

```shell
git clone https://github.com/denehoffman/gluex-ksks-paper-analysis.git
cd gluex-ksks-paper-analysis
```

2. Run via [`uv`](https://docs.astral.sh/uv/) (recommended)
```shell
uv run gluex-ksks-paper-analysis --config <config> --variation <variation>
```

Alternatively, install with `pip`:
```shell
pip install .
gluex-ksks-paper-analysis --config <config> --variation <variation>
```

> [!NOTE]
> This project is designed to run with Python 3.14 and I am unsure what the minimum supported Python version is, but it's at least 3.10 since this code uses `match` statements

## Configuration

### Configuration (`config.toml`)

The analysis is driven by a single `config.toml` file describing datasets, cuts, weights, plots, fits, and variations.
Each section defines named entries; names can be arbitrary alphanumeric strings with underscores or dashes.

---

#### **[cuts]**

Define selection rules (data which satisfies the rules will be *removed*):

```toml
[cuts.my_cut]
rules = ["m_meson > 1.0", "hx_costheta < 0.8"] # <, <=, >, and >= are supported
cache = true  # optional, store the reduced dataset in a new file if true
```

---

#### **[weights]**

Define weighting schemes:

```toml
[weights.splot_weights]
weight_type = "splot"
sig = "hist"
bkg = "exp"
cache = false # optional, store the reduced dataset in a new file if true
```

`weight_type` can be `"accidental"` or `"splot"` (the latter requires `sig` and `bkg` which can either be "hist" for a 2D histogram profile based on Monte Carlo or "exp" for a single exponential).

---

#### **[datasets]**

List input datasets and processing steps:

```toml
[datasets.data]
source = "data.parquet"
steps = ["my_cut", "splot_weights"]
```

---

#### **[plots]**

Two styles are supported:

**Histogram:**

```toml
[plots.m_meson]
variable = "m_meson"
label = "Invariant Mass"
bins = 100
limits = [1.0, 2.0]
units = "GeV"
```

**XY plot:**

```toml
[plots.xy_test]
x = "m_meson"
y = "hx_costheta" # both x and y must exist as the 1D histograms defined above
```

---

#### **[fits]**

Define amplitude fits:

```toml
[fits.fit1]
waves = ["S+0+", "D+2+"] # waves must be of the form JM or JMR where M must be signed (even M=0 must be +0). Omitting R will result in a spherical harmonic without polarization information
data = "data" # these refer to [datasets]
accmc = "accmc"
bins = 50
limits = [1.0, 2.0] # optional, [1.0, 2.0] by default
n_iterations = 20 # optional, 20 by default
n_bootstraps = 100 # optional, 100 by default
```

---

#### **[variations]**

Group related plots and fits:

```toml
[variations.nominal]
plots = [
  { dataset = "data", items = ["m_meson"] }
]
fits = ["fit1"]
```

---

#### **Notes**

* Allowed variable names for plotting and cuts are:
  + `<particle>_{e,px,py,pz}` where `<particle>` can be any of `beam`, `proton`, `kshort1`, `kshort2`, `piplus1`, `piminus1`, `piplus2`, or `piminus2`
  + Any of the following:
    - `pol_magnitude`
    - `pol_angle`
    - `ChiSqDOF`
    - `Proton_Z`
    - `m_meson`
    - `ksb_costheta`
    - `m_baryon`
    - `hx_costheta`
    - `hx_phi`
    - `mandelstam_t`
    - `reduced_mandelstam_t`
    - `m_piplus1`
    - `m_piminus1`
    - `m_piplus2`
    - `m_piminus2`
    - `m_piplus1_piminus1`
    - `m_piplus2_piminus2`
    - `m_piplus1_piminus2`
    - `m_piplus2_piminus1`
    - `m_p_piplus1`
    - `m_p_piplus2`
    - `m_p_piminus1`
    - `m_p_piminus2`
    - `m_p_piplus1_piminus1`
    - `m_p_piplus1_piminus2`
    - `m_p_piplus2_piminus1`
    - `m_p_piplus2_piminus2`
    - `m_p_piminus1_piminus2`

## TODOs
- Plots comparing data and Monte Carlo
- Plots of sPlot fit results
- Saved sPlot fits
- Reports with tables of fit values
