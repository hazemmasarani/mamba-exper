import math
import torch
from time import perf_counter
import numpy as np
from torch import nn
import torch.distributed as dist
from typing import Any, Optional, Tuple, Union
from dataclasses import dataclass
from transformers import MambaConfig
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
# from causal_conv1d import causal_conv1d_fn

class MambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MambaRMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{self.weight.shape[0]}, eps={self.variance_epsilon}"

class MambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: MambaConfig, layer_idx: int = -1):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.intermediate_size,
            padding=config.conv_kernel - 1,
        )
        self.conv1d.weight = nn.Parameter(torch.tensor(np.random.random(self.conv1d.weight.size()), dtype=self.conv1d.weight.dtype))
        

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.use_bias)
        self.in_proj.weight = nn.Parameter(torch.tensor(np.random.random(self.in_proj.weight.size()), dtype=self.in_proj.weight.dtype))
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        self.x_proj.weight = nn.Parameter(torch.tensor(np.random.random(self.x_proj.weight.size()), dtype=self.x_proj.weight.dtype))
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)
        self.dt_proj.weight = nn.Parameter(torch.tensor(np.random.random(self.dt_proj.weight.size()), dtype=self.dt_proj.weight.dtype))

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.out_proj.weight = nn.Parameter(torch.tensor(np.random.random(self.out_proj.weight.size()), dtype=self.out_proj.weight.dtype))
        self.use_bias = config.use_bias

    def slow_forward(self, input_states, attention_mask: Optional[torch.LongTensor] = None):
        # rank = self.config.rank if hasattr(self.config, 'rank') else 0
        world_size = self.config.world_size if hasattr(self.config, 'world_size') else 1
        benchmark = self.config.benchmark["block"]["mixer"]

        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # 1. Gated MLP's linear projection
        in_proj_start = perf_counter()
        projected_states = self.in_proj(input_states).transpose(1, 2)                   # [batch, 2 * intermediate_size, seq_len]
        benchmark["in_proj"].append(perf_counter() - in_proj_start)
        hidden_states, gate = projected_states.chunk(2, dim=1)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 2. Convolution sequence transformation
        ssm_state = torch.zeros(
            (batch_size, self.intermediate_size, self.ssm_state_size),
            device=hidden_states.device, dtype=dtype
        )
        conv1d_start = perf_counter()
        hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])     # [batch, intermediate_size, seq_len]
        # hidden_states = causal_conv1d_fn(hidden_states, self.conv1d.weight, self.conv1d.bias, activation=self.activation)
        benchmark["conv1d"].append(perf_counter() - conv1d_start)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        x_proj_start = perf_counter()
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        benchmark["x_proj"].append(perf_counter() - x_proj_start)

        if world_size > 1:
            # print(f'[Rank {rank}] | Distributed op start')
            grp = self.config.group if hasattr(self.config, "group") else None
            mixer_reduce_start = perf_counter()
            # ssm_parameters = ssm_parameters.to(dtype=torch.float16)
            # start = perf_counter()
            if self.config.quantize_dtype == "float16":
                dist.all_reduce(ssm_parameters.to(dtype=torch.float16), group=grp)
            else:
                dist.all_reduce(ssm_parameters, group=grp)
            # print(f'Mixer All reduce took {perf_counter() - start} seconds')
            self.config.benchmark["block"]["mixer_reduce"].append(perf_counter() - mixer_reduce_start)

            # ssm_param_tensors = [torch.zeros_like(ssm_parameters) for _ in range(world_size)]
            # dist.all_gather(ssm_param_tensors, ssm_parameters)
            # ssm_parameters = torch.zeros_like(ssm_parameters)
            # for t in ssm_param_tensors:
            #     ssm_parameters += t

            # print(f'[Rank {rank}] | Distributed op successful!', ssm_parameters.shape)
        
        ssm_parameters = ssm_parameters.to(dtype=torch.float32)
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )


        dt_proj_start = perf_counter()
        discrete_time_step = self.dt_proj(time_step)                                    # [batch, seq_len, intermediate_size]
        benchmark["dt_proj"].append(perf_counter() - dt_proj_start)
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]

        # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        discretization1_start = perf_counter()
        A = -torch.exp(self.A_log.float())                                              # [intermediate_size, ssm_state_size]
        discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()       # [batch, intermediate_size, seq_len, ssm_state_size]
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()
        benchmark["discretization1"].append(perf_counter() - discretization1_start)

        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        discretization2_start = perf_counter()
        scan_outputs = []
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediade_size, ssm_state]
            scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
            scan_outputs.append(scan_output[:, :, 0])
        scan_output = torch.stack(scan_outputs, dim=-1)                                # [batch, seq_len, intermediade_size]
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = (scan_output * self.act(gate))
        benchmark["discretization2"].append(perf_counter() - discretization2_start)

        # 4. Final linear projection
        out_proj_start = perf_counter()
        contextualized_states = self.out_proj(scan_output.transpose(1, 2))  # [batch, seq_len, hidden_size]
        benchmark["out_proj"].append(perf_counter() - out_proj_start)
        return contextualized_states
    # fmt: on

    def forward(
        self,
        hidden_states,
        cache_params: Optional[Any] = None,  # For compatability (NOT USING CACHE!!)
        cache_position: Optional[Any] = None,  # For compatability (NOT USING CACHE!!)
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        return self.slow_forward(hidden_states, attention_mask)
    
class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig, layer_idx: int = -1):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = MambaMixer(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states,
        cache_params: Optional[Any] = None,
        cache_position: Optional[Any] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        rank = self.config.rank if hasattr(self.config, 'rank') else 0
        # print(f'[Rank {rank}] | In {self.layer_idx} block (hidden_states: {hidden_states.shape})')
        residual = hidden_states
        
        rmsnorm_start = perf_counter()
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        self.config.benchmark["block"]["rmsnorm"].append(perf_counter() - rmsnorm_start)

        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        mixer_start = perf_counter()

        hidden_states = self.mixer(hidden_states, attention_mask=attention_mask)
        if rank == 0:
            hidden_states = residual + hidden_states

        self.config.benchmark["block"]["mixer"]["_total"].append(perf_counter() - mixer_start)
        return hidden_states

@dataclass
class MambaOutput(ModelOutput):
    """
    Class for the MAMBA model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`MambaCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[Any] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class MambaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MambaConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["MambaBlock", "MambaMixer"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, MambaMixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True

            dt_init_std = self.config.time_step_rank**-0.5 * self.config.time_step_scale
            if self.config.time_step_init_scheme == "constant":
                nn.init.constant_(module.dt_proj.weight, dt_init_std)
            elif self.config.time_step_init_scheme == "random":
                nn.init.uniform_(module.dt_proj.weight, -dt_init_std, dt_init_std)

            dt = torch.exp(
                torch.rand(self.config.intermediate_size)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                module.dt_proj.bias.copy_(inv_dt)
            module.dt_proj.bias._no_reinit = True

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.config.num_hidden_layers)
                        
class MambaModel(MambaPreTrainedModel):
    
    def reset_benchmark(self):
        self.config.benchmark = {
            "block": {
                "_total": [],
                "mixer": {
                    "_total": [],
                    "in_proj": [],
                    "conv1d": [],
                    "x_proj": [],
                    "discretization1": [],
                    "discretization2": [],
                    "dt_proj": [],
                    "out_proj": [],
                },
                "mixer_reduce": [],
                "rmsnorm": [],
            },
            "block_reduce": [],
            "fnorm": [],
        }
        
    def get_benchmark(self):
        return self.config.benchmark

    def __init__(self, config):
        super().__init__(config)
        
        self.reset_benchmark()

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()


    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[Any] = None,
        use_cache: Optional[bool] = False,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[Any] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MambaOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        use_cache = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache:
            if cache_params is None:
                cache_params = MambaCache(
                    self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                cache_position = torch.arange(0, self.config.conv_kernel, device=inputs_embeds.device)
            elif cache_position is None:
                # cases when we do manual forward instead of using `model.generate` which will initiate
                # `cache_position` and makes sure it is not None, throw error here instead of doing some
                # hack to conjecture the current cache position
                raise ValueError(
                    "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                    "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                    "be initialized for you automatically"
                )
        else:
            cache_params = None

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        
        world_size = self.config.world_size if hasattr(self.config, 'world_size') else 1
        rank = self.config.rank if hasattr(self.config, 'rank') else 0

        #* TEST START
        # for block in self.layers[:1]:
        #     hidden_states = block(hidden_states)
        #     if world_size > 1:
        #         dist.all_reduce(hidden_states)
        # return MambaOutput(
        #     last_hidden_state=hidden_states,
        #     cache_params=cache_params if use_cache else None,
        #     hidden_states=all_hidden_states,
        # )
        #* TEST ENDS

        layer_idx = 0
        for mixer_block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    mixer_block.__call__, hidden_states, cache_params, cache_position, attention_mask
                )
            else:
                block_start = perf_counter()
                hidden_states = mixer_block(
                    hidden_states,
                    # cache_params=cache_params,
                    # cache_position=cache_position,
                    attention_mask=attention_mask,
                )

                block_reduce_start = perf_counter()
                layer_idx += 1
                if world_size > 1:
                    # print("ALL REDUCE INSIDE MODEL")
                    grp = self.config.group if hasattr(self.config, "group") else None
                    # Block level quantized all-reduce affects accuracy
                    # if layer_idx > 15:
                    #     hidden_states = hidden_states.to(dtype=torch.float16)
                    # start = perf_counter()
                    # dist.all_reduce(hidden_states, group=grp)
                    if self.config.quantize_dtype == "float16":
                        dist.all_reduce(hidden_states.to(dtype=torch.float16), group=grp)
                    else:
                        dist.all_reduce(hidden_states, group=grp)

                    # print(f'Block {layer_idx} All reduce took {perf_counter() - start} seconds')
                    hidden_states = hidden_states.to(dtype=torch.float32)
                block_end = perf_counter()
                self.config.benchmark["block_reduce"].append(block_end - block_reduce_start)
                self.config.benchmark["block"]["_total"].append(block_end - block_start)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # print(f'[Rank {rank}] | Completed block {mixer_block.layer_idx}')
        fnorm_start = perf_counter()
        hidden_states = self.norm_f(hidden_states)
        self.config.benchmark["fnorm"].append(perf_counter() - fnorm_start)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return MambaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )