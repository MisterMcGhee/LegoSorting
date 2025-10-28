"""
chute_state_manager.py - Chute positioning state machine

This module manages the state machine for chute positioning in the sorting system.
It ensures pieces have time to fall through the chute before repositioning for
the next piece.

STATE MACHINE:
    IDLE → POSITIONED → WAITING_FOR_FALL → IDLE

RESPONSIBILITIES:
- Track chute positioning state
- Manage fall timers
- Prevent premature repositioning
- Provide chute availability status

DOES NOT:
- Control servo hardware directly (that's arduino_servo_module)
- Track bin capacity (that's bin_capacity_module)
- Find pieces to sort (that's detector/zone_manager)
"""

import time
import logging
from enum import Enum
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ChuteState(Enum):
    """
    States for chute positioning state machine.

    IDLE: Chute available, actively looking for next piece
    POSITIONED: Chute locked to specific piece, waiting for exit
    WAITING_FOR_FALL: Piece falling, chute locked, timer active
    """
    IDLE = "idle"
    POSITIONED = "positioned"
    WAITING_FOR_FALL = "waiting_for_fall"


@dataclass
class ChuteStatus:
    """
    Current status of the chute state machine.

    Used for debugging, logging, and GUI display.
    """
    state: ChuteState
    positioned_piece_id: Optional[int]
    positioned_bin: Optional[int]
    fall_time_remaining: Optional[float]
    available: bool


class ChuteStateManager:
    """
    Manages the state machine for chute positioning.

    This class ensures that pieces have sufficient time to fall through
    the chute before repositioning for the next piece. It prevents race
    conditions where the chute moves while a piece is still falling.
    """

    def __init__(self, fall_time_seconds: float = 0.5):
        """
        Initialize chute state manager.

        Args:
            fall_time_seconds: Time to wait after piece exits before
                             repositioning chute (default: 0.5s)
        """
        self.fall_time_seconds = fall_time_seconds

        # State machine variables
        self.state = ChuteState.IDLE
        self.positioned_piece_id: Optional[int] = None
        self.positioned_bin: Optional[int] = None
        self.fall_start_time: Optional[float] = None

        logger.info(f"Chute state manager initialized - Fall time: {self.fall_time_seconds}s")

    # ========================================================================
    # STATE QUERIES
    # ========================================================================

    def is_available(self) -> bool:
        """Check if chute is available for positioning."""
        available = self.state == ChuteState.IDLE
        if not available:
            logger.warning(f"Chute NOT available - current state: {self.state.value}")
        return available

    def get_current_state(self) -> ChuteState:
        """Get current state of the chute."""
        return self.state

    def get_positioned_piece_id(self) -> Optional[int]:
        """Get ID of piece chute is currently positioned for."""
        return self.positioned_piece_id

    def get_positioned_bin(self) -> Optional[int]:
        """Get bin number chute is currently positioned at."""
        return self.positioned_bin

    def get_status(self) -> ChuteStatus:
        """
        Get comprehensive status of chute state machine.

        Returns:
            ChuteStatus object with all current state information
        """
        fall_time_remaining = None

        if self.state == ChuteState.WAITING_FOR_FALL and self.fall_start_time:
            elapsed = time.time() - self.fall_start_time
            fall_time_remaining = max(0, self.fall_time_seconds - elapsed)

        return ChuteStatus(
            state=self.state,
            positioned_piece_id=self.positioned_piece_id,
            positioned_bin=self.positioned_bin,
            fall_time_remaining=fall_time_remaining,
            available=self.is_available()
        )

    # ========================================================================
    # STATE TRANSITIONS
    # ========================================================================

    def position_for_piece(self, piece_id: int, bin_number: int) -> bool:
        """
        Transition to POSITIONED state for a specific piece.

        Can only be called when state is IDLE.

        Args:
            piece_id: ID of piece to position for
            bin_number: Target bin number

        Returns:
            True if transition successful, False if invalid state
        """
        if self.state != ChuteState.IDLE:
            logger.warning(
                f"Cannot position for piece {piece_id} - "
                f"chute in {self.state.value} state"
            )
            return False

        # Transition to POSITIONED
        self.state = ChuteState.POSITIONED
        self.positioned_piece_id = piece_id
        self.positioned_bin = bin_number

        logger.info(
            f"📍 Chute positioned: Piece {piece_id} → Bin {bin_number} "
            f"(IDLE → POSITIONED)"
        )

        return True

    def notify_piece_exited(self, piece_id: int) -> bool:
        """
        Transition to WAITING_FOR_FALL state when positioned piece exits.

        Can only be called when state is POSITIONED and for the correct piece.

        Args:
            piece_id: ID of piece that exited ROI

        Returns:
            True if transition successful, False if invalid state or piece
        """
        # Must be in POSITIONED state
        if self.state != ChuteState.POSITIONED:
            logger.debug(
                f"Piece {piece_id} exited, but chute not positioned "
                f"(state: {self.state.value})"
            )
            return False

        # Must be the piece we're positioned for
        if piece_id != self.positioned_piece_id:
            logger.warning(
                f"Piece {piece_id} exited, but chute positioned for "
                f"piece {self.positioned_piece_id}"
            )
            return False

        # Transition to WAITING_FOR_FALL
        self.state = ChuteState.WAITING_FOR_FALL
        self.fall_start_time = time.time()

        logger.info(
            f"🚪 Piece {piece_id} exited → Fall timer started "
            f"({self.fall_time_seconds}s) (POSITIONED → WAITING_FOR_FALL)"
        )

        return True

    def update(self) -> None:
        """Update state machine (check fall timer expiration)."""
        if self.state != ChuteState.WAITING_FOR_FALL:
            return

        elapsed = time.time() - self.fall_start_time
        logger.debug(f"Fall timer: {elapsed:.2f}s / {self.fall_time_seconds}s")

        if elapsed >= self.fall_time_seconds:
            logger.info(
                f"✅ Fall complete ({elapsed:.2f}s) → Chute available "
                f"(WAITING_FOR_FALL → IDLE)"
            )
            self.state = ChuteState.IDLE
            self.positioned_piece_id = None
            self.positioned_bin = None
            self.fall_start_time = None

    # ========================================================================
    # RESET AND CLEANUP
    # ========================================================================

    def reset(self) -> None:
        """
        Reset state machine to IDLE.

        Used for error recovery or system reset.
        """
        logger.info("Resetting chute state machine to IDLE")

        self.state = ChuteState.IDLE
        self.positioned_piece_id = None
        self.positioned_bin = None
        self.fall_start_time = None

    def __repr__(self) -> str:
        """String representation for debugging."""
        if self.state == ChuteState.IDLE:
            return "ChuteStateManager(IDLE)"
        elif self.state == ChuteState.POSITIONED:
            return f"ChuteStateManager(POSITIONED: piece={self.positioned_piece_id}, bin={self.positioned_bin})"
        else:  # WAITING_FOR_FALL
            elapsed = time.time() - self.fall_start_time if self.fall_start_time else 0
            remaining = max(0, self.fall_time_seconds - elapsed)
            return f"ChuteStateManager(WAITING_FOR_FALL: {remaining:.2f}s remaining)"


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_chute_state_manager(config_manager) -> ChuteStateManager:
    """
    Factory function to create ChuteStateManager instance.

    Args:
        config_manager: Enhanced configuration manager

    Returns:
        Initialized ChuteStateManager instance
    """
    arduino_config = config_manager.get_module_config("arduino_servo")
    fall_time_seconds = arduino_config.get("fall_time_seconds", 0.5)

    return ChuteStateManager(fall_time_seconds=fall_time_seconds)