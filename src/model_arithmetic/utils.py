from loguru import logger

ENABLE_LOGGING = False

def enable_logging():
    """
    Enables logging by setting the global variable ENABLE_LOGGING to True.
    """
    global ENABLE_LOGGING
    ENABLE_LOGGING = True

def get_max_length(model_config, default_length=1024):
    """
    Get the maximum length from the model configuration.

    Args:
        model_config (object): The model configuration object.
        default_length (int, optional): The default maximum length. Defaults to 1024.

    Returns:
        int: The maximum length.
    """
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model_config, length_setting, None)
        if max_length:
            if ENABLE_LOGGING:
                logger.debug(f"Found max length: {max_length}")
            break
    if not max_length:
        max_length = default_length
        if ENABLE_LOGGING:
            logger.debug(f"Using default max length: {max_length}")

    return max_length

def log(function, message):
    """
    Logs the given message using the provided function if logging is enabled.
    
    Parameters:
        function (callable): The logging function to use.
        message (str): The message to be logged.
    """
    if ENABLE_LOGGING:
        function(message)