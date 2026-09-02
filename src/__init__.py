"""
This file registers the model with the Python SDK.
"""

from viam.services.vision import Vision
from viam.resource.registry import Registry, ResourceCreatorRegistration

from .claude import claude


Registry.register_resource_creator(Vision.API, claude.MODEL, ResourceCreatorRegistration(claude.new, claude.validate))
