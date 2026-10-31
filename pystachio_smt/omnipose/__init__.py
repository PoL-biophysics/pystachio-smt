from . import core, utils, loss, plot, misc, cli, data, gpu
import pkg_resources
try:
    __version__ = pkg_resources.get_distribution("omnipose").version
except Exception:
    __version__ = "1.0.0"