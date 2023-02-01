name_to_block = {}
name_to_input = {}

import src.moe.server.layers.albert
import src.moe.server.layers.common
import src.moe.server.layers.dropout
from src.moe.server.layers.custom_experts import add_custom_models_from_file, register_expert_class
from src.moe.server.layers.lr_schedule import get_linear_schedule_with_warmup

schedule_name_to_scheduler = {"linear": get_linear_schedule_with_warmup, "none": None}
