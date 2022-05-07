from typing import TYPE_CHECKING


from monai.config import IgniteInfo
from monai.handlers.stats_handler import StatsHandler
from monai.utils import is_scalar, min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")

DEFAULT_KEY_VAL_FORMAT = "{}: {:.4f} "
DEFAULT_TAG = "Loss"


class StatsHandlerEx(StatsHandler):
    def exception_raised(self, engine: Engine, e: Exception) -> None:
        """
        Handler for train or validation/evaluation exception raised Event.
        Print the exception information and traceback. This callback may be skipped because the logic
        with Ignite can only trigger the first attached handler for `EXCEPTION_RAISED` event.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            e: the exception caught in Ignite during engine.run().

        """
        # self.logger.exception(f"Exception: {e}")
        raise e