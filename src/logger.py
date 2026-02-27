"""TensorBoard metric logging.

Writes per-update training metrics (losses, learning rate, throughput)
to TensorBoard and prints a progress summary to stdout.
"""


def log_metrics(writer, optimizer, metrics, global_step, sps, update, num_updates):
    """Write training metrics to TensorBoard and print progress.

    Args:
        writer: TensorBoard SummaryWriter.
        optimizer: The optimizer (used to read current learning rate).
        metrics: Dict of loss metrics from ppo_update().
        global_step: Total environment steps taken so far.
        sps: Steps per second (throughput).
        update: Current update number.
        num_updates: Total number of updates planned.
    """
    writer.add_scalar(
        "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
    )
    writer.add_scalar("charts/SPS", sps, global_step)
    for key, value in metrics.items():
        writer.add_scalar(f"losses/{key}", value, global_step)
    print(f"SPS: {sps}, update: {update}/{num_updates}")
