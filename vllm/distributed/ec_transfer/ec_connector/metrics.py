# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import Any, Union


@dataclass
class ECConnectorStats:
    """
    Base class for EC Connector Stats, a container for transfer performance 
    metrics or otherwise important telemetry from the connector. 
    All sub-classes need to be serializable as stats are sent from worker to
    logger process.
    """
    data: dict[str, Any] = field(default_factory=dict)

    def reset(self):
        """Reset the stats, clear the state."""
        raise NotImplementedError

    def aggregate(self, other: "ECConnectorStats") -> "ECConnectorStats":
        """
        Aggregate stats with another `ECConnectorStats` object.
        """
        raise NotImplementedError

    def reduce(self) -> dict[str, Union[int, float]]:
        """
        Reduce the observations collected during a time interval to one or 
        more representative values (eg avg/median/sum of the series). 
        This is meant to be called by the logger to produce a summary of the
        stats for the last time interval.
        """
        raise NotImplementedError

    def is_empty(self) -> bool:
        """Return True if the stats are empty."""
        raise NotImplementedError
