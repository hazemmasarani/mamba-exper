from transformers import MambaConfig

def make_split_config(config: MambaConfig, world: int):
    return MambaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size // world,
        state_size=config.state_size,
        num_hidden_layers=config.num_hidden_layers,
        layer_norm_epsilon=config.layer_norm_epsilon,
        pad_token_id=config.pad_token_id,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        expand=config.expand,
        conv_kernel=config.conv_kernel,
        use_bias=config.use_bias,
        use_conv_bias=config.use_conv_bias,
        hidden_act=config.hidden_act,
        initializer_range=config.initializer_range,
        residual_in_fp32=config.residual_in_fp32,
        time_step_rank=config.time_step_rank,
        time_step_scale=config.time_step_scale,
        time_step_min=config.time_step_min,
        time_step_max=config.time_step_max,
        time_step_init_scheme=config.time_step_init_scheme,
        time_step_floor=config.time_step_floor,
        rescale_prenorm_residual=config.rescale_prenorm_residual,
        use_cache=config.use_cache,
        use_mambapy=config.use_mambapy,
    )