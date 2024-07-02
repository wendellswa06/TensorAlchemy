def get_miner_version() -> str:
    """Returns version of miner (i.e. 1.0.1)"""
    import neurons.miner

    return neurons.miner.__version__


def get_miner_spec_version() -> int:
    """Returns numeric representation of miner's version (i.e. 10001)"""
    import neurons.miner

    return neurons.miner.__spec_version__
