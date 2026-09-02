from typing import ClassVar, Mapping, Optional, Any, List, cast
from typing_extensions import Self
import base64
import json
import math
import os
import re
from io import BytesIO

from viam.proto.common import PointCloudObject
from viam.proto.service.vision import Classification, Detection
from viam.utils import ValueTypes

from viam.module.types import Reconfigurable
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily

from viam.services.vision import Vision, CaptureAllResult
from viam.proto.service.vision import GetPropertiesResponse
from viam.components.camera import Camera, ViamImage
from viam.media.utils.pil import viam_to_pil_image
from viam.media.video import CameraMimeType
from viam.logging import getLogger

from anthropic import AsyncAnthropic
from PIL import Image as PILImage

LOGGER = getLogger(__name__)

DEFAULT_CLASSIFICATION_PROMPT = "describe this image"
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_THINKING_BUDGET = 2048

# Claude standard-resolution vision tier limits.
# See https://platform.claude.com/docs/en/build-with-claude/vision-coordinates
STANDARD_MAX_EDGE = 1568
STANDARD_MAX_TOKENS = 1568


def count_image_tokens(width: int, height: int) -> int:
    """Visual tokens consumed by an image: one token per 28x28 pixel patch."""
    return math.ceil(width / 28) * math.ceil(height / 28)


def resized_size(
    width: int,
    height: int,
    max_edge: int = STANDARD_MAX_EDGE,
    max_tokens: int = STANDARD_MAX_TOKENS,
) -> tuple:
    """The size Claude resizes an image to before padding."""

    def fits(w: int, h: int) -> bool:
        return (
            math.ceil(w / 28) * 28 <= max_edge
            and math.ceil(h / 28) * 28 <= max_edge
            and count_image_tokens(w, h) <= max_tokens
        )

    if fits(width, height):
        return (width, height)
    if height > width:
        resized_h, resized_w = resized_size(height, width, max_edge, max_tokens)
        return (resized_w, resized_h)

    aspect_ratio = width / height
    lo, hi = 1, width
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid, max(round(mid / aspect_ratio), 1)):
            lo = mid
        else:
            hi = mid
    return (lo, max(round(lo / aspect_ratio), 1))


def prepare_image(pil_image: PILImage.Image) -> PILImage.Image:
    """Resize so Claude's pixel coordinates map 1:1 onto the image we send."""
    rgb = pil_image.convert("RGB")
    target = resized_size(*rgb.size)
    if target == rgb.size:
        return rgb
    return rgb.resize(target, PILImage.Resampling.LANCZOS)


def pil_to_base64_jpeg(pil_image: PILImage.Image) -> str:
    buf = BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def extract_json(text: str) -> Any:
    """Parse JSON from a model response, tolerating optional markdown fences."""
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
    if fence:
        stripped = fence.group(1).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", stripped)
        if not match:
            raise
        return json.loads(match.group(1))


class claude(Vision, Reconfigurable):
    """
    Vision service backed by Anthropic Claude cloud models.
    """

    MODEL: ClassVar[Model] = Model(ModelFamily("mcvella", "vision"), "claude")

    client: AsyncAnthropic
    model_name: str
    max_tokens: int
    classification_prompt: str
    reasoning: bool
    DEPS: Mapping[ResourceName, ResourceBase]

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        my_class = cls(config.name)
        my_class.reconfigure(config, dependencies)
        return my_class

    @classmethod
    def validate(cls, config: ComponentConfig):
        fields = config.attributes.fields
        api_key = fields["api_key"].string_value or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise Exception("api_key is required (set attributes.api_key or ANTHROPIC_API_KEY)")
        camera = fields["camera"].string_value
        if not camera:
            raise Exception("camera is required")
        return [camera], []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        self.DEPS = dependencies
        fields = config.attributes.fields

        api_key = fields["api_key"].string_value or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise Exception("api_key is required (set attributes.api_key or ANTHROPIC_API_KEY)")

        camera_name = fields["camera"].string_value
        if Camera.get_resource_name(camera_name) not in dependencies:
            raise Exception(f"camera dependency '{camera_name}' not found")

        self.classification_prompt = (
            fields["classification_prompt"].string_value or DEFAULT_CLASSIFICATION_PROMPT
        )

        self.reasoning = False
        if "reasoning" in fields:
            self.reasoning = fields["reasoning"].bool_value

        self.model_name = fields["model"].string_value or DEFAULT_MODEL

        self.max_tokens = DEFAULT_MAX_TOKENS
        if "max_tokens" in fields and fields["max_tokens"].number_value:
            self.max_tokens = int(fields["max_tokens"].number_value)

        workspace_id = (
            fields["workspace_id"].string_value
            or os.environ.get("ANTHROPIC_WORKSPACE_ID", "")
        )

        client_kwargs: dict = {"api_key": api_key}
        if workspace_id:
            client_kwargs["default_headers"] = {
                "anthropic-workspace-id": workspace_id
            }

        LOGGER.info(
            f"initializing Claude vision with model={self.model_name}"
            + (f", workspace_id={workspace_id}" if workspace_id else "")
        )
        self.client = AsyncAnthropic(**client_kwargs)

    async def get_cam_image(self, camera_name: str) -> ViamImage:
        cam = cast(Camera, self.DEPS[Camera.get_resource_name(camera_name)])
        images, _ = await cam.get_images()
        if not images:
            raise Exception("get_images from cam returned no images")
        for img in images:
            if img.mime_type == CameraMimeType.JPEG:
                return img
        raise Exception(f"no images from cam is {CameraMimeType.JPEG}")

    def _resolve_reasoning(self, extra: Optional[Mapping[str, Any]] = None) -> bool:
        if extra is not None and "reasoning" in extra:
            return bool(extra["reasoning"])
        return self.reasoning

    def _object_names_from_text(self, text: str) -> List[str]:
        return [obj.strip() for obj in str(text).split(",") if obj.strip()]

    def _object_list_prompt(self, query: Optional[str] = None) -> str:
        if query and str(query).strip():
            return (
                f"List all {str(query).strip()} you can see in this image. "
                "Return your answer as a simple comma-separated list of object names. "
                "Do not include any other text."
            )
        return (
            "List all the objects you can see in this image. "
            "Return your answer as a simple comma-separated list of object names. "
            "Do not include any other text."
        )

    def _detection_prompt(self, object_names: List[str], width: int, height: int) -> str:
        names = ", ".join(object_names)
        return (
            f"Locate each of these objects in the image: {names}.\n"
            f"The image is {width}x{height} pixels. The origin (0, 0) is the top-left corner; "
            "x increases to the right and y increases downward.\n"
            "Return ONLY a JSON array. Each element must be an object with keys "
            '"class_name" (string) and "bbox" ([x_min, y_min, x_max, y_max] as absolute pixel integers).\n'
            "Include one entry per visible instance. Omit objects that are not present. "
            "Do not wrap the JSON in markdown."
        )

    async def _query(
        self,
        pil_image: PILImage.Image,
        prompt: str,
        *,
        reasoning: bool = False,
        max_tokens: Optional[int] = None,
    ) -> str:
        image = prepare_image(pil_image)
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": pil_to_base64_jpeg(image),
                },
            },
            {"type": "text", "text": prompt},
        ]

        tokens = max_tokens or self.max_tokens
        kwargs: dict = {
            "model": self.model_name,
            "max_tokens": tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if reasoning:
            budget = min(DEFAULT_THINKING_BUDGET, max(1024, tokens // 2))
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            kwargs["max_tokens"] = max(tokens, budget + 1024)

        response = await self.client.messages.create(**kwargs)
        parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "\n".join(parts).strip()

    async def _list_objects(
        self,
        pil_image: PILImage.Image,
        query: Optional[str] = None,
        *,
        reasoning: bool = False,
    ) -> List[str]:
        answer = await self._query(
            pil_image, self._object_list_prompt(query), reasoning=reasoning
        )
        return self._object_names_from_text(answer)

    async def _detect_objects(
        self, pil_image: PILImage.Image, object_names: List[str]
    ) -> List[Detection]:
        if not object_names:
            return []

        original_width, original_height = pil_image.size
        view = prepare_image(pil_image)
        view_width, view_height = view.size

        answer = await self._query(
            pil_image,
            self._detection_prompt(object_names, view_width, view_height),
            max_tokens=max(self.max_tokens, 2048),
        )
        try:
            parsed = extract_json(answer)
        except (json.JSONDecodeError, ValueError) as exc:
            LOGGER.warning(f"failed to parse detection JSON: {exc}; raw={answer!r}")
            return []

        if isinstance(parsed, dict):
            parsed = parsed.get("objects") or parsed.get("detections") or []
        if not isinstance(parsed, list):
            LOGGER.warning(f"unexpected detection payload type: {type(parsed)}")
            return []

        scale_x = original_width / view_width if view_width else 1.0
        scale_y = original_height / view_height if view_height else 1.0
        detections = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            class_name = str(item.get("class_name") or item.get("label") or "").strip()
            bbox = item.get("bbox") or item.get("box")
            if not class_name or not bbox or len(bbox) != 4:
                # Also accept flat x_min/y_min/x_max/y_max fields
                if all(k in item for k in ("x_min", "y_min", "x_max", "y_max")):
                    bbox = [item["x_min"], item["y_min"], item["x_max"], item["y_max"]]
                else:
                    continue
            try:
                x_min_v, y_min_v, x_max_v, y_max_v = [float(v) for v in bbox]
            except (TypeError, ValueError):
                continue

            x_min = int(round(x_min_v * scale_x))
            y_min = int(round(y_min_v * scale_y))
            x_max = int(round(x_max_v * scale_x))
            y_max = int(round(y_max_v * scale_y))

            x_min = max(0, min(x_min, original_width))
            x_max = max(0, min(x_max, original_width))
            y_min = max(0, min(y_min, original_height))
            y_max = max(0, min(y_max, original_height))
            if x_max <= x_min or y_max <= y_min:
                continue

            detections.append({
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "x_min_normalized": x_min / original_width if original_width else 0.0,
                "y_min_normalized": y_min / original_height if original_height else 0.0,
                "x_max_normalized": x_max / original_width if original_width else 0.0,
                "y_max_normalized": y_max / original_height if original_height else 0.0,
                "confidence": 1,
                "class_name": class_name,
            })
        return detections

    async def get_detections_from_camera(
        self, camera_name: str, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None
    ) -> List[Detection]:
        return await self.get_detections(await self.get_cam_image(camera_name), extra=extra)

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        # Auto-label: query for object names, then ask Claude for bounding boxes.
        # Pass extra={"query": "..."} to limit which objects are listed.
        # Pass extra={"objects": "a, b"} or a list to skip the listing query.
        pil_image = viam_to_pil_image(image)
        object_names: List[str] = []
        if extra is not None and extra.get("objects") is not None:
            objects = extra["objects"]
            if isinstance(objects, str):
                object_names = self._object_names_from_text(objects)
            else:
                object_names = [str(obj).strip() for obj in objects if str(obj).strip()]
        else:
            query = extra.get("query") if extra else None
            object_names = await self._list_objects(
                pil_image, query, reasoning=self._resolve_reasoning(extra)
            )
        return await self._detect_objects(pil_image, object_names)

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        return await self.get_classifications(await self.get_cam_image(camera_name), count, extra=extra)

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        question = self.classification_prompt
        if extra is not None and extra.get("question") is not None:
            question = extra["question"]
        result = await self._query(
            viam_to_pil_image(image),
            question,
            reasoning=self._resolve_reasoning(extra),
        )
        return [{"class_name": result, "confidence": 1}]

    async def get_object_point_clouds(
        self, camera_name: str, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None
    ) -> List[PointCloudObject]:
        return

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None) -> Mapping[str, ValueTypes]:
        return

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        result = CaptureAllResult()
        result.image = await self.get_cam_image(camera_name)
        # Classifications use classification_prompt; detections use a separate
        # object-listing query. These prompts differ, so do not reuse results.
        if return_classifications:
            result.classifications = await self.get_classifications(
                result.image, 1, extra=extra
            )
        if return_detections:
            result.detections = await self.get_detections(result.image, extra=extra)
        return result

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> GetPropertiesResponse:
        return GetPropertiesResponse(
            classifications_supported=True,
            detections_supported=True,
            object_point_clouds_supported=False,
        )
