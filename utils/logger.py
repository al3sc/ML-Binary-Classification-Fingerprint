import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import atexit

class Logger:
    def __init__(self, file_name, output_dir="./assets/log", lvl="DEBUG", resume=False):
        """
        Initialize the logger and create a log file (if not exists).
        If the file already exists and resume is True, the log file is re-opened.
        
        Args:
            filename (str): Log file name.
            level (int): Logging level (logging.INFO, logging.DEBUG, ...).
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = f"{output_dir}/{file_name}.log"

        if os.path.exists(file_path) and not resume:
            print(f"ERROR: The log file '{file_path}' already exists. Aborting execution.")
            sys.exit(1)
        
        self.logger = logging.getLogger(file_path)
        self.logger.setLevel(self.__getLoggingLevel__(lvl))

        # Configure handler only once
        if not self.logger.handlers:
            file_handler = RotatingFileHandler(file_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
            #formatter = logging.Formatter('%(levelname)s - %(message)s')
            formatter = CustomFormatter('%(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Register the close function to be called at program termination
        atexit.register(self.__close__)

        if not resume:
            self.log("Logger initialized", lvl="INFO")
        else:
            self.log_separator()
            self.log("Logger resumed", lvl="WARNING")


    def __getLoggingLevel__(self, level):
        match level:
            case "INFO":
                return logging.INFO
            case "WARNING":
                return logging.WARNING
            case "ERROR":
                return logging.ERROR
            case "CRITICAL":
                return logging.CRITICAL
            case _:
                return logging.DEBUG

    def log(self, message="", lvl="DEBUG"):
        """
        Write a message in the log file.
        
        Args:
            message (str): Message to be logged.
            level (int): Logging level (logging.INFO, logging.DEBUG, ...).
        """
        match lvl:
            case "INFO":
                self.logger.info(message)
            case "WARNING":
                self.logger.warning(message)
            case "ERROR":
                self.logger.error(message)
            case "CRITICAL":
                self.logger.critical(message)
            case _:
                self.logger.debug(message)
        self.logger.handlers[0].flush()

    def info(self, message=""):
        """
        Write an info message in the log file.
        
        Args:
            message (str): Info message to be logged.
        """
        self.logger.info(message)
        self.logger.handlers[0].flush()
    
    def error(self, message=""):
        """
        Write an error message in the log file.
        
        Args:
            message (str): Error message to be logged.
        """
        self.logger.critical(message)
        self.logger.handlers[0].flush()
    

    def log_title(self, title):
        self.log_separator(double=True)
        self.log(f"{title.upper()} ######\n", lvl="INFO")

    def log_paragraph(self, title):
        self.log_separator(separator="-")
        self.log(f"#### {title.upper()} ####\n")
    
    def log_separator(self, separator="=", double=False):
        """Prints a separator line in the log."""
        self.log(f"\n{separator * 50 * (2 if double else 1)}\n")

    def __close__(self):
        """Close the logger's file handler to ensure the log is saved and free memory."""
        self.log_separator(double=True)
        self.log("Logger closed correctly!", lvl="INFO")
        self.logger.handlers[0].close()


class CustomFormatter(logging.Formatter):
    def format(self, record):
        # If the log message is already formatted, return it directly
        if hasattr(record, "formatted_message"):
            return record.formatted_message

        # Add timestamp only for INFO, WARNING, and CRITICAL
        if record.levelname in ["INFO", "WARNING", "CRITICAL"]:
            record.formatted_message = f"{record.levelname} - {self.formatTime(record, '%Y-%m-%d %H:%M:%S')} - {record.msg}"
            return record.formatted_message
        
        return super().format(record)
    


# Usage example
if __name__ == "__main__":
    logger = Logger("app")
    logger.log_title("LOG FILE EXAMPLE")
    logger.log(f"{'\u03B3'}")
#     logger.log("Debug message example")
#     logger.log_empty_line()
#     logger.log("A warning message", lvl="WARNING")
#     logger.log("A critical error", lvl="CRITICAL")
#     logger.log_separator()
#     logger.log("Last line")
#     logger.log_separator()
#     logger.log_title("Title")
#     logger.log_paragraph("Paragraph")
