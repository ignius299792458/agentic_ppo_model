def log_metrics(writer, optimizer, metrics, global_step, sps, update, num_updates):
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("charts/SPS", sps, global_step)
    for key, value in metrics.items():
        writer.add_scalar(f"losses/{key}", value, global_step)
    print(f"SPS: {sps}, update: {update}/{num_updates}")
