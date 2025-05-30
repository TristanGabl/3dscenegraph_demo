import logging

class CustomColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[93m",  # Orange
        "INFO": "\033[92m",   # Green
        "ERROR": "\033[91m",  # Red
    }
    RESET = "\033[0m"
    
    def format(self, record):
        message_color = self.COLORS.get(record.levelname, self.RESET)
        record.asctime = f"{self.formatTime(record, self.datefmt)}"
        formatted_message = f"{message_color}[{record.asctime}-{record.levelname}] {self.RESET}{record.msg}"        
        return formatted_message
    
def setup_logger(DEBUG_: bool = False):

    formatter = CustomColorFormatter(datefmt='%H:%M:%S')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger("colored_logger")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    def important(self, message):
        self.log(logging.INFO, "\033[95m#----\033[0m")
        self.log(logging.INFO, f"\033[95m# {message}\033[0m")
        self.log(logging.INFO, "\033[95m#----\033[0m")

    logging.Logger.important = important

    if DEBUG_:
        logger.debug("DEBUG: True")
    return logger