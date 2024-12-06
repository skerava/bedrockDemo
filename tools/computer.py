import json
import logging
import sys
import os

import asyncio
import shlex
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dependency.base import BaseAnthropicTool, ToolError, ToolResult
from dependency.run import run

from Quartz.CoreGraphics import CGDisplayBounds, CGMainDisplayID
from anthropic.types.beta import BetaToolComputerUse20241022Param

import json

OUTPUT_DIR = "/tmp/outputs"
KEY_MAPPING_JSON = os.path.join(os.path.dirname(__file__), '..', 'config', 'cliclick_key_mapping.json')
TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
]


class Resolution(TypedDict):
    width: int
    height: int


# sizes above XGA/WXGA are not recommended (see README.md)
# scale down to one of these targets if ComputerTool._scaling_enabled is set
MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
}


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None


def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]

def get_screen_resolution():
    main_display_id = CGMainDisplayID()
    main_display_bounds = CGDisplayBounds(main_display_id)
    width = int(main_display_bounds.size.width)
    height = int(main_display_bounds.size.height)
    return width, height

class ComputerTool(BaseAnthropicTool):
    """
    A tool that allows the agent to interact with the screen, keyboard, and mouse of the current computer.
    The tool parameters are defined by Anthropic and are not editable.
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int
    display_num: int | None

    _screenshot_delay = 2.0
    _scaling_enabled = True

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(
            ScalingSource.COMPUTER, self.width, self.height
        )
        return {
            "display_width_px": width,
            "display_height_px": height,
            "display_number": self.display_num,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    def __init__(self):
        super().__init__()

        self.width, self.height = get_screen_resolution()
        self.display_num = 1
        assert self.width and self.height, "WIDTH, HEIGHT must be set"
        if (display_num := os.getenv("DISPLAY_NUM")) is not None:
            self.display_num = int(display_num)
            self._display_prefix = f"DISPLAY=:{self.display_num} "
        else:
            self.display_num = None
            self._display_prefix = ""

    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
    ):
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if not isinstance(coordinate, list) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) and i >= 0 for i in coordinate):
                raise ToolError(f"{coordinate} must be a tuple of non-negative ints")

            x, y = self.scale_coordinates(
                ScalingSource.API, coordinate[0], coordinate[1]
            )

            if action == "mouse_move":
                logging.info(f"Moving mouse to {x}, {y}")
                try:
                    return await self.shell(f"cliclick m:{x},{y}")
                except Exception as e:
                    logging.error(f"Error moving mouse: {e}")
                    return ToolResult(error=str(e))
            elif action == "left_click_drag":
                logging.info(f"Dragging mouse to {x}, {y}")
                try:
                    return await self.shell(f"cliclick dd:. du:{x},{y}")
                except Exception as e:
                    logging.error(f"Error dragging mouse: {e}")
                    return ToolResult(error=str(e))
                
        elif action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(output=f"{text} must be a string")

            if action == "key":
                config_path = KEY_MAPPING_JSON
                logging.info(f"Reading key mapping from {config_path}")
                with open(config_path, "r") as f:
                    key_mapping = json.load(f)
                cmd = key_mapping.get(text, text)
                try:
                    logging.info(f"Pressing key: cliclick {cmd}")
                    return await self.shell(f"cliclick {cmd}")
                except Exception as e:
                    logging.error(f"Error pressing key: {e}")
                    return ToolResult(error=str(e))
            elif action == "type":
                results: list[ToolResult] = []
                for chunk in chunks(text, TYPING_GROUP_SIZE):
                    cmd = f"cliclick t:{shlex.quote(chunk)}"
                    results.append(await self.shell(cmd, take_screenshot=False))
                screenshot_image = (await self.screenshot()).image_bytes
                return ToolResult(
                    output="".join(result.output or "" for result in results),
                    error="".join(result.error or "" for result in results),
                    image_bytes=screenshot_image,
                )
        elif action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
            "screenshot",
            "cursor_position",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            if action == "screenshot":
                return await self.screenshot()
            elif action == "cursor_position":
                result = await self.shell(
                    f"cliclick p:",
                    take_screenshot=False,
                )
                output = result.output or ""
                x, y = map(int, output.strip().split(","))
                x, y = self.scale_coordinates(
                    ScalingSource.COMPUTER,
                    x,
                    y,
                )
                return result.replace(output=f"X={x},Y={y}")
            else:
                click_arg = {
                    "left_click": "c:.",
                    "right_click": "rc:.",
                    "middle_click": "mc:.",
                    "double_click": "dc:.",
                }[action]
                logging.info(f"Performing cliclick {click_arg}")
                return await self.shell(f"cliclick {click_arg}")
        else:
            raise ToolError(f"Invalid action: {action}")

    async def screenshot(self):
        """Take a screenshot of the current screen and return the image."""
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"screenshot_{uuid4().hex}.png"

        # Try gnome-screenshot first
        screenshot_cmd = f"{self._display_prefix}screencapture {path}"

        result = await self.shell(screenshot_cmd, take_screenshot=False)
        if self._scaling_enabled:
            x, y = self.scale_coordinates(
                ScalingSource.COMPUTER, self.width, self.height
            )
            try:  
                await self.shell(
                    f"convert {path} -resize {x}x{y}! {path}", take_screenshot=False
                )
                logging.info(f"Screenshot resized to {x}x{y}")
            except Exception as e:
                logging.error(f"Error resizing screenshot: {e}")

        if path.exists():
            with open(path, "rb") as image_file:
                image_bytes = image_file.read()
            return result.replace(image_bytes = image_bytes)
        raise ToolError(f"Failed to take screenshot: {result.error}")

    async def shell(self, command: str, take_screenshot=True) -> ToolResult:
        """Run a shell command and return the output, error, and optionally a screenshot."""
        _, stdout, stderr = await run(command)
        image_bytes = None

        if take_screenshot:
            # delay to let things settle before taking a screenshot
            await asyncio.sleep(self._screenshot_delay)
            image_bytes = (await self.screenshot()).image_bytes

        return ToolResult(output=stdout, error=stderr, image_bytes=image_bytes)

    def scale_coordinates(self, source: ScalingSource, x: int, y: int):
        """Scale coordinates to a target maximum resolution."""
        if not self._scaling_enabled:
            return x, y
        ratio = self.width / self.height
        target_dimension = None
        for dimension in MAX_SCALING_TARGETS.values():
            # allow some error in the aspect ratio - not ratios are exactly 16:9
            if abs(dimension["width"] / dimension["height"] - ratio) < 0.02:
                if dimension["width"] < self.width:
                    target_dimension = dimension
                break
        if target_dimension is None:
            return x, y
        # should be less than 1
        x_scaling_factor = target_dimension["width"] / self.width
        y_scaling_factor = target_dimension["height"] / self.height
        if source == ScalingSource.API:
            if x > self.width or y > self.height:
                raise ToolError(f"Coordinates {x}, {y} are out of bounds")
            # scale up
            return round(x / x_scaling_factor), round(y / y_scaling_factor)
        # scale down
        return round(x * x_scaling_factor), round(y * y_scaling_factor)
    
def invoke(input_data):
    logging.info(f"input_data: {input_data}")
    try:
        logging.info(f"input_data: {input_data}")
        computer = ComputerTool()
        response = asyncio.run(computer(**input_data))
        tool_result_content = {}
        if response.error:
            tool_result_content["json"]["Error"] = response.error
        if response.output:
            tool_result_content["json"]["Output"] = response.output
        if response.image_bytes:
            tool_result_content["image"] = {
                    "format":"png",
                    "source":{
                    "bytes": response.image_bytes
                    }
               }
            
        return tool_result_content
    except Exception as e:
        logging.error(f"Error: {e}")
        return {"json":{"error": type(e).__name__, "message": str(e)}}
    
    