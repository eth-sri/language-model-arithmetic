from loguru import logger

ENABLE_LOGGING = True

def enable_logging():
    global ENABLE_LOGGING
    ENABLE_LOGGING = True

def get_max_length(model_config, default_length=1024):
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model_config, length_setting, None)
        if max_length:
            break
    if not max_length:
        max_length = default_length
        if ENABLE_LOGGING:
            logger.debug(f"Max length not found. Using default max length: {max_length}")

    return max_length

def log(function, message):
    if ENABLE_LOGGING:
        function(message)