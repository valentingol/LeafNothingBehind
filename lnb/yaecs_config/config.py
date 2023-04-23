"""Project configuration class."""
import random
from typing import Optional

from yaecs import Configuration


def random_run_id(run_id: Optional[int]) -> int:
    """Return a random run id."""
    return run_id or random.randint(0, 999999)  # return random run id if None


class ProjectConfig(Configuration):
    """Project configuration class."""

    @staticmethod
    def get_default_config_path() -> str:
        """Return default config."""
        return "config/default.yaml"

    def parameters_pre_processing(self) -> dict:
        """Pre-processing on some config parameters."""
        return {"run_id": random_run_id}

    def parameters_post_processing(self) -> dict:
        """Post-processing on some config parameters."""
        return {}
