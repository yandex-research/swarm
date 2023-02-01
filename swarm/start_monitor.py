import time
from typing import List

import src
import wandb
from src import choose_ip_address
from src.utils.logging import TextStyle, get_logger, use_src_log_handler
from multiaddr import Multiaddr
from transformers import HfArgumentParser

from arguments import TrainingMonitorArguments
from utils import LocalMetrics

use_src_log_handler("in_root_logger")
logger = get_logger(__name__)


def log_visible_maddrs(visible_maddrs: List[Multiaddr], only_p2p: bool) -> None:
    if only_p2p:
        unique_addrs = {addr["p2p"] for addr in visible_maddrs}
        initial_peers_str = " ".join(f"/p2p/{addr}" for addr in unique_addrs)
    else:
        available_ips = [Multiaddr(addr) for addr in visible_maddrs if "ip4" in addr or "ip6" in addr]
        if available_ips:
            preferred_ip = choose_ip_address(available_ips)
            selected_maddrs = [addr for addr in visible_maddrs if preferred_ip in str(addr)]
        else:
            selected_maddrs = visible_maddrs
        initial_peers_str = " ".join(str(addr) for addr in selected_maddrs)

    logger.info(
        f"Running a DHT peer. To connect other peers to this one over the Internet, use "
        f"{TextStyle.BOLD}{TextStyle.BLUE}--initial_peers {initial_peers_str}{TextStyle.RESET}"
    )
    logger.info(f"Full list of visible multiaddresses: {' '.join(str(addr) for addr in visible_maddrs)}")


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingMonitorArguments,))
    monitor_args, = parser.parse_args_into_dataclasses()

    experiment_prefix = monitor_args.experiment_prefix

    dht = src.DHT(
        start=True,
        initial_peers=monitor_args.initial_peers,
        host_maddrs=monitor_args.host_maddrs,
        announce_maddrs=monitor_args.announce_maddrs,
    )
    log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=False)

    if monitor_args.wandb_project is not None:
        wandb.init(project=monitor_args.wandb_project)

    current_step = 0

    while True:
        metrics_dict = dht.get(experiment_prefix + "_metrics", latest=True)
        if metrics_dict is not None:
            metrics_dict = metrics_dict.value
            metrics = [LocalMetrics.parse_obj(metrics_dict[peer].value) for peer in metrics_dict]
            if metrics:
                latest_step = max(item.step for item in metrics)

                if latest_step != current_step:
                    logger.debug(f"Got metrics from {len(metrics)} peers")

                    for i, metrics_for_peer in enumerate(metrics):
                        logger.debug(f"{i} peer {metrics_for_peer}")

                    current_step = latest_step
                    alive_peers = 0
                    sum_loss = 0

                    for item in metrics:
                        sum_loss += item.loss
                        alive_peers += 1
                    current_loss = sum_loss / alive_peers
                    logger.info(f"Step #{current_step}\tloss = {current_loss:.5f}")

                    if monitor_args.wandb_project is not None:
                        wandb.log(
                            {
                                "loss": current_loss,
                                "alive peers": alive_peers,
                                "step": latest_step,
                            }
                        )
        time.sleep(monitor_args.refresh_period)
