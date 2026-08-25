"""Armin says hi — animated XBM easter egg. Mirrors armin.ts."""
from __future__ import annotations

import random
import threading
from typing import Any, Literal

from pi_coding_agent.modes.interactive.theme.theme import theme

from .component import Component

WIDTH = 31
HEIGHT = 36
BITS = [
    0xFF, 0xFF, 0xFF, 0x7F, 0xFF, 0xF0, 0xFF, 0x7F, 0xFF, 0xED, 0xFF, 0x7F, 0xFF, 0xDB, 0xFF, 0x7F, 0xFF, 0xB7, 0xFF,
    0x7F, 0xFF, 0x77, 0xFE, 0x7F, 0x3F, 0xF8, 0xFE, 0x7F, 0xDF, 0xFF, 0xFE, 0x7F, 0xDF, 0x3F, 0xFC, 0x7F, 0x9F, 0xC3,
    0xFB, 0x7F, 0x6F, 0xFC, 0xF4, 0x7F, 0xF7, 0x0F, 0xF7, 0x7F, 0xF7, 0xFF, 0xF7, 0x7F, 0xF7, 0xFF, 0xE3, 0x7F, 0xF7,
    0x07, 0xE8, 0x7F, 0xEF, 0xF8, 0x67, 0x70, 0x0F, 0xFF, 0xBB, 0x6F, 0xF1, 0x00, 0xD0, 0x5B, 0xFD, 0x3F, 0xEC, 0x53,
    0xC1, 0xFF, 0xEF, 0x57, 0x9F, 0xFD, 0xEE, 0x5F, 0x9F, 0xFC, 0xAE, 0x5F, 0x1F, 0x78, 0xAC, 0x5F, 0x3F, 0x00, 0x50,
    0x6C, 0x7F, 0x00, 0xDC, 0x77, 0xFF, 0xC0, 0x3F, 0x78, 0xFF, 0x01, 0xF8, 0x7F, 0xFF, 0x03, 0x9C, 0x78, 0xFF, 0x07,
    0x8C, 0x7C, 0xFF, 0x0F, 0xCE, 0x78, 0xFF, 0xFF, 0xCF, 0x7F, 0xFF, 0xFF, 0xCF, 0x78, 0xFF, 0xFF, 0xDF, 0x78, 0xFF,
    0xFF, 0xDF, 0x7D, 0xFF, 0xFF, 0x3F, 0x7E, 0xFF, 0xFF, 0xFF, 0x7F,
]
BYTES_PER_ROW = (WIDTH + 7) // 8
DISPLAY_HEIGHT = (HEIGHT + 1) // 2
Effect = Literal["typewriter", "scanline", "rain", "fade", "crt", "glitch", "dissolve"]
EFFECTS: tuple[Effect, ...] = ("typewriter", "scanline", "rain", "fade", "crt", "glitch", "dissolve")


def get_pixel(x: int, y: int) -> bool:
    if y >= HEIGHT:
        return False
    byte_index = y * BYTES_PER_ROW + (x // 8)
    return ((BITS[byte_index] >> (x % 8)) & 1) == 0


def get_char(x: int, row: int) -> str:
    upper = get_pixel(x, row * 2)
    lower = get_pixel(x, row * 2 + 1)
    if upper and lower:
        return "█"
    if upper:
        return "▀"
    if lower:
        return "▄"
    return " "


def build_final_grid() -> list[list[str]]:
    return [[get_char(x, row) for x in range(WIDTH)] for row in range(DISPLAY_HEIGHT)]


class ArminComponent(Component):
    name = "armin"

    def __init__(self, ui: Any | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ui = ui
        self.effect: Effect = random.choice(EFFECTS)
        self.final_grid = build_final_grid()
        self.current_grid = self._empty_grid()
        self.effect_state: dict[str, Any] = {}
        self.cached_lines: list[str] = []
        self.cached_width = 0
        self.grid_version = 0
        self.cached_version = -1
        self._timer: threading.Timer | None = None
        self._init_effect()
        self._start_animation()

    def invalidate(self) -> None:
        self.cached_width = 0
        super().invalidate()

    def render(self, width: int = 80) -> list[str]:
        if width == self.cached_width and self.cached_version == self.grid_version:
            return self.cached_lines
        padding = 1
        available = width - padding
        lines = []
        for row in self.current_grid:
            clipped = "".join(row[:available])
            pad_right = max(0, width - padding - len(clipped))
            lines.append(f" {theme.fg('accent', clipped)}{' ' * pad_right}")
        message = "ARMIN SAYS HI"
        msg_pad = max(0, width - padding - len(message))
        lines.append(f" {theme.fg('accent', message)}{' ' * msg_pad}")
        self.cached_lines = lines
        self.cached_width = width
        self.cached_version = self.grid_version
        return lines

    def _empty_grid(self) -> list[list[str]]:
        return [[" " for _ in range(WIDTH)] for _ in range(DISPLAY_HEIGHT)]

    def _init_effect(self) -> None:
        if self.effect == "typewriter":
            self.effect_state = {"pos": 0}
        elif self.effect == "scanline":
            self.effect_state = {"row": 0}
        elif self.effect == "rain":
            self.effect_state = {
                "drops": [{"y": -random.randint(0, DISPLAY_HEIGHT * 2 - 1), "settled": 0} for _ in range(WIDTH)]
            }
        elif self.effect == "fade":
            positions = [(row, x) for row in range(DISPLAY_HEIGHT) for x in range(WIDTH)]
            random.shuffle(positions)
            self.effect_state = {"positions": positions, "idx": 0}
        elif self.effect == "crt":
            self.effect_state = {"expansion": 0}
        elif self.effect == "glitch":
            self.effect_state = {"phase": 0, "glitchFrames": 8}
        else:
            chars = [" ", "░", "▒", "▓", "█", "▀", "▄"]
            self.current_grid = [[random.choice(chars) for _ in range(WIDTH)] for _ in range(DISPLAY_HEIGHT)]
            positions = [(row, x) for row in range(DISPLAY_HEIGHT) for x in range(WIDTH)]
            random.shuffle(positions)
            self.effect_state = {"positions": positions, "idx": 0}

    def _start_animation(self) -> None:
        fps = 60 if self.effect == "glitch" else 30
        delay = 1 / fps

        def tick() -> None:
            done = self._tick_effect()
            self.grid_version += 1
            if self.ui is not None and hasattr(self.ui, "request_render"):
                self.ui.request_render()
            if done:
                self.stop_animation()
                return
            self._timer = threading.Timer(delay, tick)
            self._timer.daemon = True
            self._timer.start()

        self._timer = threading.Timer(delay, tick)
        self._timer.daemon = True
        self._timer.start()

    def stop_animation(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _tick_effect(self) -> bool:
        if self.effect == "typewriter":
            return self._tick_typewriter()
        if self.effect == "scanline":
            return self._tick_scanline()
        if self.effect == "rain":
            return self._tick_rain()
        if self.effect == "fade":
            return self._tick_fade()
        if self.effect == "crt":
            return self._tick_crt()
        if self.effect == "glitch":
            return self._tick_glitch()
        return self._tick_dissolve()

    def _tick_typewriter(self) -> bool:
        state = self.effect_state
        for _ in range(3):
            row = state["pos"] // WIDTH
            x = state["pos"] % WIDTH
            if row >= DISPLAY_HEIGHT:
                return True
            self.current_grid[row][x] = self.final_grid[row][x]
            state["pos"] += 1
        return False

    def _tick_scanline(self) -> bool:
        state = self.effect_state
        if state["row"] >= DISPLAY_HEIGHT:
            return True
        self.current_grid[state["row"]] = list(self.final_grid[state["row"]])
        state["row"] += 1
        return False

    def _tick_rain(self) -> bool:
        state = self.effect_state
        all_settled = True
        self.current_grid = self._empty_grid()
        for x in range(WIDTH):
            drop = state["drops"][x]
            for row in range(DISPLAY_HEIGHT - 1, DISPLAY_HEIGHT - drop["settled"] - 1, -1):
                if row >= 0:
                    self.current_grid[row][x] = self.final_grid[row][x]
            if drop["settled"] >= DISPLAY_HEIGHT:
                continue
            all_settled = False
            target_row = -1
            for row in range(DISPLAY_HEIGHT - 1 - drop["settled"], -1, -1):
                if self.final_grid[row][x] != " ":
                    target_row = row
                    break
            drop["y"] += 1
            if 0 <= drop["y"] < DISPLAY_HEIGHT:
                if target_row >= 0 and drop["y"] >= target_row:
                    drop["settled"] = DISPLAY_HEIGHT - target_row
                    drop["y"] = -random.randint(1, 5)
                else:
                    self.current_grid[drop["y"]][x] = "▓"
        return all_settled

    def _tick_fade(self) -> bool:
        state = self.effect_state
        for _ in range(15):
            if state["idx"] >= len(state["positions"]):
                return True
            row, x = state["positions"][state["idx"]]
            self.current_grid[row][x] = self.final_grid[row][x]
            state["idx"] += 1
        return False

    def _tick_crt(self) -> bool:
        state = self.effect_state
        mid = DISPLAY_HEIGHT // 2
        self.current_grid = self._empty_grid()
        top = mid - state["expansion"]
        bottom = mid + state["expansion"]
        for row in range(max(0, top), min(DISPLAY_HEIGHT - 1, bottom) + 1):
            self.current_grid[row] = list(self.final_grid[row])
        state["expansion"] += 1
        return state["expansion"] > DISPLAY_HEIGHT

    def _tick_glitch(self) -> bool:
        state = self.effect_state
        if state["phase"] < state["glitchFrames"]:
            rows = []
            for row in self.final_grid:
                offset = random.randint(-3, 3)
                glitch_row = list(row)
                if random.random() < 0.3:
                    glitch_row = glitch_row[offset:] + glitch_row[:offset]
                    glitch_row = glitch_row[:WIDTH]
                elif random.random() < 0.2:
                    glitch_row = list(self.final_grid[random.randrange(DISPLAY_HEIGHT)])
                rows.append(glitch_row)
            self.current_grid = rows
            state["phase"] += 1
            return False
        self.current_grid = [list(row) for row in self.final_grid]
        return True

    def _tick_dissolve(self) -> bool:
        state = self.effect_state
        for _ in range(20):
            if state["idx"] >= len(state["positions"]):
                return True
            row, x = state["positions"][state["idx"]]
            self.current_grid[row][x] = self.final_grid[row][x]
            state["idx"] += 1
        return False

    def dispose(self) -> None:
        self.stop_animation()
