"""Physical rank and parameter-cost contracts for the GPTQ entrypoint."""


def factor_ranks(
    full_name: str,
    shape: tuple[int, int],
    *,
    base_pac_rank: int,
    ogc_rank: int,
    pac_increment: int = 0,
) -> tuple[int, int]:
    """Return the component ranks to fit for this saved linear map."""
    if any(type(rank) is not int or rank < 0 for rank in (base_pac_rank, ogc_rank, pac_increment)):
        raise ValueError(f"{full_name}: PAC floor, increment and OGC rank must be nonnegative integers")
    pac_rank = base_pac_rank + pac_increment
    if pac_rank + ogc_rank == 0:
        raise ValueError(f"{full_name}: GPTQModel LoRA requires a positive total rank")
    if max(pac_rank, ogc_rank) > min(shape):
        raise ValueError(f"{full_name}: component ranks exceed the matrix dimensions {shape}")
    return pac_rank, ogc_rank
