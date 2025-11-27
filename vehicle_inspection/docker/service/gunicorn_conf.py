"""Gunicorn configuration."""
import multiprocessing

# https://docs.gunicorn.org/en/stable/design.html#how-many-workers
max_workers = 12

# Restart gunicorn workers after 10000 requests to limit the damage
# of potential memory leaks
max_requests = 10000

# Add some randomness to prevent from stopping gunicorn workers
# at the same time
max_requests_jitter = 500

# Use 2 * the number of CPUs, plus 1, as recommended for Gunicorn workers
workers = min(multiprocessing.cpu_count() * 2 + 1, max_workers)

# Set the worker class to "gevent" for efficient I/O handling
worker_class = "gevent"

# Listen on port 8080, which is the default port for Cloud Run
bind = "0.0.0.0:8080"

# Log to standard output and standard error
accesslog = "-"
errorlog = "-"

# Enable graceful shutdown
graceful_timeout = 120

# Timeout of request process
timeout = 30

# Use the default Gunicorn logging configuration
logconfig_dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "root": {"level": "INFO", "handlers": ["console"]},
    "loggers": {
        "gunicorn.error": {"level": "INFO", "handlers": ["error_console"], "propagate": False, "qualname": "gunicorn.error"},
        "gunicorn.access": {"level": "INFO", "handlers": ["console"], "propagate": False, "qualname": "gunicorn.access"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "generic", "stream": "ext://sys.stdout"},
        "error_console": {"class": "logging.StreamHandler", "formatter": "generic", "stream": "ext://sys.stderr"},
    },
    "formatters": {
        "generic": {"format": "%(asctime)s [%(process)d] [%(levelname)s] %(message)s", "datefmt": "[%Y-%m-%d %H:%M:%S %z]", "class": "logging.Formatter"}
    },
}
