from .chatterbox_node import NODE_CLASS_MAPPINGS as BASE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BASE_DISPLAY_NAME_MAPPINGS
from .chatterbox_dialog_node import NODE_CLASS_MAPPINGS as DIALOG_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as DIALOG_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(BASE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(DIALOG_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(BASE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(DIALOG_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]