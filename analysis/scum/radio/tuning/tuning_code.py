"""The tuning code represents the coarse, mid, and fine code settings on SCuM.

Each of the coarse, mid, and fine codes are 5 bits wide and controls the
corresponding capacitive DAC of SCuM's LO. The tuning code is a 15-bit number.
"""

import numpy as np

# Number of bits in the coarse, mid, and fine codes.
NUM_COARSE_MID_FINE_BITS = 5

# Number of bits in a tuning code.
NUM_TUNING_CODE_BITS = 15


class TuningCode:
    """Represents a tuning code consisting of a coarse, mid, and fine code."""

    def __init__(
        self,
        coarse: int = None,
        mid: int = None,
        fine: int = None,
        *,
        tuning_code: int = None,
    ):
        if tuning_code is not None:
            (
                self.coarse,
                self.mid,
                self.fine,
            ) = self.tuning_code_to_coarse_mid_fine(tuning_code)
        else:
            self.coarse = coarse
            self.mid = mid
            self.fine = fine

    @property
    def tuning_code(self) -> int:
        """Tuning code."""
        return self.coarse_mid_fine_to_tuning_code(
            self.coarse,
            self.mid,
            self.fine,
        )

    @staticmethod
    def coarse_mid_fine_to_tuning_code(
            coarse: int | np.ndarray, mid: int | np.ndarray,
            fine: int | np.ndarray) -> int | np.ndarray:
        """Transforms the coarse, mid, and fine code(s) to the tuning code(s).

        Args:
            coarse: Coarse code(s).
            mid: Mid code(s).
            fine: Fine code(s).

        Returns:
            The tuning code(s).
        """
        return (coarse * 2**(2 * NUM_COARSE_MID_FINE_BITS) +
                mid * 2**NUM_COARSE_MID_FINE_BITS + fine)

    @staticmethod
    def tuning_code_to_coarse_mid_fine(
        tuning_code: int | np.ndarray
    ) -> tuple[int | np.ndarray, int | np.ndarray, int | np.ndarray]:
        """Transforms the tuning code(s) to the coarse, mid, and fine code(s).

        Args:
            tuning_code: Tuning code(s).

        Returns:
            The coarse, mid, and fine code(s).
        """
        return (
            (tuning_code >> 10) & 0x1F,
            (tuning_code >> 5) & 0x1F,
            tuning_code & 0x1F,
        )
