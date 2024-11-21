import logging


def setup_logging():
    """
    Sets up logging configuration globally for the package.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_class_logger(cls):
    """
    Factory function to get a logger with the full path of the class.
    """
    logger_name = f"{cls.__module__}.{cls.__name__}"
    logger = logging.getLogger(logger_name)
    return logger
