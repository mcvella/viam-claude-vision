# claude modular vision service

This module implements the [rdk vision API](https://github.com/rdk/vision-api) in a mcvella:vision:claude model.

This model uses [Anthropic Claude](https://docs.anthropic.com/) cloud vision models for image classification, querying, and object detection. An [Anthropic API key](https://console.anthropic.com/settings/keys) is required.

## Build and Run

To use this module, follow these instructions to [add a module from the Viam Registry](https://docs.viam.com/registry/configure/#add-a-modular-resource-from-the-viam-registry) and select the `mcvella:vision:claude` model from the [mcvella claude-vision module](https://app.viam.com/module/mcvella/claude-vision).

For local development, you can also add this module from a local path or git URL in your machine config.

## Configure your vision service

> [!NOTE]
> Before configuring your vision service, you must [create a machine](https://docs.viam.com/manage/fleet/machines/#add-a-new-machine).

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/).
Click on the **Service** subtab and click **Create service**.
Select the `vision` type, then select the `mcvella:vision:claude` model.
Enter a name for your vision service and click **Create**.

On the new service panel, copy and paste the following attribute template into your vision service's **Attributes** box:

```json
{
  "api_key": "<your Anthropic API key>",
  "camera": "<camera-name>"
}
```

> [!NOTE]
> For more information, see [Configure a Robot](https://docs.viam.com/manage/configuration/).

### Attributes

The following attributes are available for `mcvella:vision:claude` model:

| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `api_key` | string | **Required** | Anthropic API key from [console.anthropic.com](https://console.anthropic.com/settings/keys). Can also be supplied via the `ANTHROPIC_API_KEY` environment variable. |
| `camera` | string | **Required** | Default camera dependency for the service. Camera-based API methods use the `camera_name` argument; add extra cameras via `depends_on` if needed. |
| `model` | string | Optional | Claude model ID to use. Defaults to `claude-sonnet-4-6`. Examples: `claude-haiku-4-5`, `claude-opus-4-6`, `claude-sonnet-5`. |
| `classification_prompt` | string | Optional | Default question for classifications. Defaults to `"describe this image"`. Overridden by `extra.question` when provided. |
| `reasoning` | bool | Optional | Enable Claude [extended thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking) for higher-quality answers (`false` by default; adds latency and cost). Overridden by `extra.reasoning` when provided. |
| `max_tokens` | number | Optional | Max output tokens for Claude responses. Defaults to `1024`. |

### Example Configurations

Default (Claude Sonnet):

```json
{
  "api_key": "YOUR_API_KEY",
  "camera": "cam"
}
```

Faster / cheaper Haiku model:

```json
{
  "api_key": "YOUR_API_KEY",
  "camera": "cam",
  "model": "claude-haiku-4-5"
}
```

Custom classification prompt with reasoning:

```json
{
  "api_key": "YOUR_API_KEY",
  "camera": "cam",
  "classification_prompt": "what safety gear is visible?",
  "reasoning": true
}
```

## API

The claude resource provides the following methods from Viam's built-in [rdk:service:vision API](https://python.viam.dev/autoapi/viam/services/vision/client/index.html)

Camera-based methods use the `camera_name` argument. That camera must be available as a dependency (the required `camera` attribute, and any additional cameras listed in `depends_on`).

### get_classifications(image=*binary*, count)

### get_classifications_from_camera(camera_name=*string*, count)

By default, the Claude model is asked the configured `classification_prompt` (or `"describe this image"` if unset).
Override per call with the extra parameter `question`. Enable extended thinking via config `reasoning` or `extra.reasoning`:

```python
claude.get_classifications(
    image,
    1,
    extra={"question": "what is the person wearing?", "reasoning": True},
)
```

### get_detections(image=*binary*)

### get_detections_from_camera(camera_name=*string*)

Detections use an automatic labeling flow: query the image for a comma-separated list of object names, then ask Claude for bounding boxes for each name.

By default, all visible objects are listed and detected. Pass `extra={"query": "..."}` to limit the list (for example, only people or vehicles), or `extra={"objects": "cup, bowl"}` to skip listing and detect specific names:

```python
claude.get_detections(image, extra={"query": "people"})
claude.get_detections(image, extra={"objects": "person, helmet"})
```

Bounding boxes are returned in absolute pixels (and normalized 0–1) relative to the original image. Claude is prompted for pixel coordinates after the image is resized to match Claude's vision preprocessing, per [Anthropic's coordinate guidance](https://platform.claude.com/docs/en/build-with-claude/vision-coordinates).
