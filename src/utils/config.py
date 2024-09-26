from pathlib import Path, PosixPath
from omegaconf import OmegaConf, DictConfig


def get_config(exp: str, config_dir: Path = PosixPath('../config')) -> DictConfig:
    OmegaConf.register_new_resolver("path", lambda x: Path(x), replace=True)
    global_config = OmegaConf.load(config_dir / 'global.yml')
    exp_config = OmegaConf.load(config_dir / f'exp_{exp}.yml')
    config = OmegaConf.merge(global_config, exp_config)
    
    config.output_path = config.output_path / exp
    config.oof_path = config.oof_path / exp
    config.output_path.mkdir(exist_ok=True, parents=True)
    config.oof_path.mkdir(exist_ok=True, parents=True)
    return config