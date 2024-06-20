import logging
import sys

# Your existing logger setup
logger = logging.getLogger('log')
logger.setLevel(logging.NOTSET)  # Default level

def setup_logging(log_filename, log_level):
    # Set the logging level based on the provided log_level
    logger.setLevel(log_level)
    #logger.propagate = False
    # Create a file handler for writing logs to a file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(log_level)

    # Create a stream handler for printing logs to the terminal
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

# Example usage
#setup_logging('app.log', logging.INFO)
logger.info("This is an info message")
