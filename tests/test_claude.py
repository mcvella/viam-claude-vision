from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from viam.components.camera import Camera
from viam.media.video import CameraMimeType, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from src.claude import claude as Claude, extract_json, resized_size


def make_config(attrs: dict, name: str = "claude") -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def make_jpeg_image(width: int = 100, height: int = 50) -> ViamImage:
    buf = BytesIO()
    Image.new("RGB", (width, height), color="red").save(buf, format="JPEG")
    return ViamImage(data=buf.getvalue(), mime_type=CameraMimeType.JPEG)


def make_camera(image: ViamImage | None = None) -> MagicMock:
    cam = MagicMock(spec=Camera)
    cam.get_images = AsyncMock(return_value=([image or make_jpeg_image()], None))
    return cam


def make_text_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock()
    with patch("src.claude.AsyncAnthropic", return_value=client) as ctor:
        yield ctor, client


@pytest.fixture
def service(mock_client):
    _, client = mock_client
    cam = make_camera()
    deps = {Camera.get_resource_name("cam"): cam}
    config = make_config({"api_key": "test-key", "camera": "cam"})
    instance = Claude.new(config, deps)
    instance._test_camera = cam
    instance._test_client = client
    return instance


class TestHelpers:
    def test_resized_size_a4_example(self):
        assert resized_size(1075, 1520) == (924, 1307)

    def test_extract_json_raw(self):
        assert extract_json('[{"class_name": "cup", "bbox": [1, 2, 3, 4]}]')[0]["class_name"] == "cup"

    def test_extract_json_fenced(self):
        text = 'Here you go:\n```json\n[{"class_name": "bowl", "bbox": [0, 0, 10, 10]}]\n```'
        assert extract_json(text)[0]["class_name"] == "bowl"


class TestValidate:
    def test_requires_api_key(self):
        with pytest.raises(Exception, match="api_key is required"):
            Claude.validate(make_config({"camera": "cam"}))

    def test_requires_camera(self):
        with pytest.raises(Exception, match="camera is required"):
            Claude.validate(make_config({"api_key": "test-key"}))

    def test_returns_camera_dependency(self):
        assert Claude.validate(make_config({"api_key": "test-key", "camera": "cam"})) == (
            ["cam"],
            [],
        )

    def test_accepts_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        assert Claude.validate(make_config({"camera": "cam"})) == (["cam"], [])


class TestReconfigure:
    def test_defaults_model(self, mock_client):
        ctor, _ = mock_client
        cam = make_camera()
        service = Claude.new(
            make_config({"api_key": "test-key", "camera": "cam"}),
            {Camera.get_resource_name("cam"): cam},
        )
        ctor.assert_called_once_with(api_key="test-key")
        assert service.model_name == "claude-sonnet-4-6"
        assert service.reasoning is False

    def test_passes_model_and_max_tokens(self, mock_client):
        cam = make_camera()
        service = Claude.new(
            make_config(
                {
                    "api_key": "test-key",
                    "camera": "cam",
                    "model": "claude-haiku-4-5",
                    "max_tokens": 2048,
                    "reasoning": True,
                }
            ),
            {Camera.get_resource_name("cam"): cam},
        )
        assert service.model_name == "claude-haiku-4-5"
        assert service.max_tokens == 2048
        assert service.reasoning is True

    def test_uses_env_api_key(self, mock_client, monkeypatch):
        ctor, _ = mock_client
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        cam = make_camera()
        Claude.new(
            make_config({"camera": "cam"}),
            {Camera.get_resource_name("cam"): cam},
        )
        ctor.assert_called_once_with(api_key="env-key")

    def test_passes_workspace_id(self, mock_client):
        ctor, _ = mock_client
        cam = make_camera()
        Claude.new(
            make_config(
                {
                    "api_key": "test-key",
                    "camera": "cam",
                    "workspace_id": "wrkspc_123",
                }
            ),
            {Camera.get_resource_name("cam"): cam},
        )
        ctor.assert_called_once_with(
            api_key="test-key",
            default_headers={"anthropic-workspace-id": "wrkspc_123"},
        )

    def test_uses_env_workspace_id(self, mock_client, monkeypatch):
        ctor, _ = mock_client
        monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_env")
        cam = make_camera()
        Claude.new(
            make_config({"api_key": "test-key", "camera": "cam"}),
            {Camera.get_resource_name("cam"): cam},
        )
        ctor.assert_called_once_with(
            api_key="test-key",
            default_headers={"anthropic-workspace-id": "wrkspc_env"},
        )


class TestClassifications:
    @pytest.mark.asyncio
    async def test_default_question(self, service):
        service._test_client.messages.create.return_value = make_text_response("a red square")
        result = await service.get_classifications(make_jpeg_image(), 1)
        assert result == [{"class_name": "a red square", "confidence": 1}]
        kwargs = service._test_client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-4-6"
        assert "thinking" not in kwargs
        user_content = kwargs["messages"][0]["content"]
        assert user_content[1]["text"] == "describe this image"
        assert user_content[0]["type"] == "image"

    @pytest.mark.asyncio
    async def test_config_classification_prompt(self, mock_client):
        _, client = mock_client
        client.messages.create.return_value = make_text_response("a helmet")
        cam = make_camera()
        service = Claude.new(
            make_config(
                {
                    "api_key": "test-key",
                    "camera": "cam",
                    "classification_prompt": "what safety gear is visible?",
                }
            ),
            {Camera.get_resource_name("cam"): cam},
        )

        await service.get_classifications(make_jpeg_image(), 1)

        prompt = client.messages.create.call_args.kwargs["messages"][0]["content"][1]["text"]
        assert prompt == "what safety gear is visible?"

    @pytest.mark.asyncio
    async def test_extra_question_overrides_config_prompt(self, mock_client):
        _, client = mock_client
        client.messages.create.return_value = make_text_response("yes")
        cam = make_camera()
        service = Claude.new(
            make_config(
                {
                    "api_key": "test-key",
                    "camera": "cam",
                    "classification_prompt": "what safety gear is visible?",
                }
            ),
            {Camera.get_resource_name("cam"): cam},
        )

        result = await service.get_classifications(
            make_jpeg_image(), 1, extra={"question": "is there a person?"}
        )

        assert result[0]["class_name"] == "yes"
        prompt = client.messages.create.call_args.kwargs["messages"][0]["content"][1]["text"]
        assert prompt == "is there a person?"

    @pytest.mark.asyncio
    async def test_config_reasoning(self, mock_client):
        _, client = mock_client
        client.messages.create.return_value = make_text_response("detailed")
        cam = make_camera()
        service = Claude.new(
            make_config(
                {
                    "api_key": "test-key",
                    "camera": "cam",
                    "reasoning": True,
                }
            ),
            {Camera.get_resource_name("cam"): cam},
        )

        await service.get_classifications(make_jpeg_image(), 1)

        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["thinking"]["type"] == "enabled"

    @pytest.mark.asyncio
    async def test_extra_reasoning_overrides_config(self, mock_client):
        _, client = mock_client
        client.messages.create.return_value = make_text_response("quick")
        cam = make_camera()
        service = Claude.new(
            make_config(
                {
                    "api_key": "test-key",
                    "camera": "cam",
                    "reasoning": True,
                }
            ),
            {Camera.get_resource_name("cam"): cam},
        )

        await service.get_classifications(
            make_jpeg_image(), 1, extra={"reasoning": False}
        )

        assert "thinking" not in client.messages.create.call_args.kwargs

    @pytest.mark.asyncio
    async def test_from_camera(self, service):
        service._test_client.messages.create.return_value = make_text_response("from cam")
        result = await service.get_classifications_from_camera("cam", 1)
        assert result[0]["class_name"] == "from cam"
        service._test_camera.get_images.assert_awaited()

    @pytest.mark.asyncio
    async def test_from_camera_uses_requested_camera(self, mock_client):
        _, client = mock_client
        client.messages.create.return_value = make_text_response("from cam-b")
        cam_a = make_camera()
        cam_b = make_camera()
        deps = {
            Camera.get_resource_name("cam-a"): cam_a,
            Camera.get_resource_name("cam-b"): cam_b,
        }
        service = Claude.new(
            make_config({"api_key": "test-key", "camera": "cam-a"}),
            deps,
        )

        await service.get_classifications_from_camera("cam-b", 1)

        cam_b.get_images.assert_awaited()
        cam_a.get_images.assert_not_awaited()


class TestDetections:
    @pytest.mark.asyncio
    async def test_auto_label_all_objects(self, service):
        service._test_client.messages.create.side_effect = [
            make_text_response("person, chair"),
            make_text_response(
                '[{"class_name": "person", "bbox": [10, 10, 30, 20]},'
                ' {"class_name": "chair", "bbox": [50, 25, 90, 45]}]'
            ),
        ]

        result = await service.get_detections(make_jpeg_image(100, 50))

        assert service._test_client.messages.create.await_count == 2
        list_prompt = service._test_client.messages.create.call_args_list[0].kwargs[
            "messages"
        ][0]["content"][1]["text"]
        assert "List all the objects you can see" in list_prompt
        assert [d["class_name"] for d in result] == ["person", "chair"]
        assert result[0]["x_min"] == 10
        assert result[0]["y_min"] == 10
        assert result[0]["x_max"] == 30
        assert result[0]["y_max"] == 20
        assert result[0]["x_min_normalized"] == pytest.approx(0.1)
        assert result[1]["x_min"] == 50
        assert result[1]["y_max"] == 45

    @pytest.mark.asyncio
    async def test_query_limits_object_list(self, service):
        service._test_client.messages.create.side_effect = [
            make_text_response("person"),
            make_text_response('[{"class_name": "person", "bbox": [0, 0, 100, 50]}]'),
        ]

        result = await service.get_detections(
            make_jpeg_image(), extra={"query": "people"}
        )

        prompt = service._test_client.messages.create.call_args_list[0].kwargs[
            "messages"
        ][0]["content"][1]["text"]
        assert "List all people you can see" in prompt
        assert result[0]["class_name"] == "person"

    @pytest.mark.asyncio
    async def test_detection_query_respects_reasoning_extra(self, service):
        service._test_client.messages.create.side_effect = [
            make_text_response("person"),
            make_text_response('[{"class_name": "person", "bbox": [0, 0, 50, 50]}]'),
        ]

        await service.get_detections(make_jpeg_image(), extra={"reasoning": True})

        list_kwargs = service._test_client.messages.create.call_args_list[0].kwargs
        assert list_kwargs["thinking"]["type"] == "enabled"

    @pytest.mark.asyncio
    async def test_objects_extra_skips_listing_query(self, service):
        service._test_client.messages.create.return_value = make_text_response(
            '[{"class_name": "cup", "bbox": [0, 0, 50, 50]},'
            ' {"class_name": "bowl", "bbox": [50, 0, 100, 50]}]'
        )

        result = await service.get_detections(
            make_jpeg_image(), extra={"objects": "cup, bowl"}
        )

        assert [d["class_name"] for d in result] == ["cup", "bowl"]
        assert service._test_client.messages.create.await_count == 1

    @pytest.mark.asyncio
    async def test_empty_object_list(self, service):
        service._test_client.messages.create.return_value = make_text_response("")
        result = await service.get_detections(make_jpeg_image())
        assert result == []
        assert service._test_client.messages.create.await_count == 1

    @pytest.mark.asyncio
    async def test_from_camera(self, service):
        service._test_client.messages.create.side_effect = [
            make_text_response("cup"),
            make_text_response('[{"class_name": "cup", "bbox": [0, 0, 50, 25]}]'),
        ]
        result = await service.get_detections_from_camera("cam")
        assert len(result) == 1
        service._test_camera.get_images.assert_awaited()

    @pytest.mark.asyncio
    async def test_from_camera_uses_requested_camera(self, mock_client):
        _, client = mock_client
        client.messages.create.side_effect = [
            make_text_response("cup"),
            make_text_response('[{"class_name": "cup", "bbox": [0, 0, 50, 25]}]'),
        ]
        cam_a = make_camera()
        cam_b = make_camera()
        service = Claude.new(
            make_config({"api_key": "test-key", "camera": "cam-a"}),
            {
                Camera.get_resource_name("cam-a"): cam_a,
                Camera.get_resource_name("cam-b"): cam_b,
            },
        )

        await service.get_detections_from_camera("cam-b")

        cam_b.get_images.assert_awaited()
        cam_a.get_images.assert_not_awaited()


class TestPropertiesAndCaptureAll:
    @pytest.mark.asyncio
    async def test_properties(self, service):
        props = await service.get_properties()
        assert props.classifications_supported is True
        assert props.detections_supported is True
        assert props.object_point_clouds_supported is False

    @pytest.mark.asyncio
    async def test_capture_all_respects_flags(self, service):
        service._test_client.messages.create.side_effect = [
            make_text_response("a red helmet on a table"),
            make_text_response("person, chair"),
            make_text_response(
                '[{"class_name": "person", "bbox": [0, 0, 50, 25]},'
                ' {"class_name": "chair", "bbox": [50, 25, 100, 50]}]'
            ),
        ]

        result = await service.capture_all_from_camera(
            "cam",
            return_image=True,
            return_classifications=True,
            return_detections=True,
        )

        assert result.image is not None
        assert result.classifications[0]["class_name"] == "a red helmet on a table"
        assert [d["class_name"] for d in result.detections] == ["person", "chair"]
        assert service._test_client.messages.create.await_count == 3
        prompts = [
            call.kwargs["messages"][0]["content"][1]["text"]
            for call in service._test_client.messages.create.call_args_list
        ]
        assert prompts[0] == "describe this image"
        assert "List all the objects you can see" in prompts[1]

    @pytest.mark.asyncio
    async def test_capture_all_classifications_only_uses_classification_prompt(self, service):
        service._test_client.messages.create.return_value = make_text_response(
            "a red helmet on a table"
        )

        result = await service.capture_all_from_camera(
            "cam",
            return_classifications=True,
        )

        assert result.classifications[0]["class_name"] == "a red helmet on a table"
        prompt = service._test_client.messages.create.call_args.kwargs["messages"][0][
            "content"
        ][1]["text"]
        assert prompt == "describe this image"
        assert service._test_client.messages.create.await_count == 1

    @pytest.mark.asyncio
    async def test_capture_all_detections_only_still_lists_objects(self, service):
        service._test_client.messages.create.side_effect = [
            make_text_response("box"),
            make_text_response('[{"class_name": "box", "bbox": [0, 0, 100, 50]}]'),
        ]

        result = await service.capture_all_from_camera(
            "cam",
            return_detections=True,
        )

        assert result.detections[0]["class_name"] == "box"
        assert service._test_client.messages.create.await_count == 2

    @pytest.mark.asyncio
    async def test_capture_all_skips_unrequested(self, service):
        result = await service.capture_all_from_camera("cam")
        assert result.image is not None
        assert not result.classifications
        assert not result.detections
        service._test_client.messages.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_capture_all_uses_requested_camera(self, mock_client):
        cam_a = make_camera()
        cam_b = make_camera()
        service = Claude.new(
            make_config({"api_key": "test-key", "camera": "cam-a"}),
            {
                Camera.get_resource_name("cam-a"): cam_a,
                Camera.get_resource_name("cam-b"): cam_b,
            },
        )

        await service.capture_all_from_camera("cam-b")

        cam_b.get_images.assert_awaited()
        cam_a.get_images.assert_not_awaited()
