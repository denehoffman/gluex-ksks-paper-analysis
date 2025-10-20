default:
    @just --list

run VARIATION="default" CONFIG="config.toml":
    @uv run gluex-ksks-paper-analysis --config {{CONFIG}} --variation {{VARIATION}}
