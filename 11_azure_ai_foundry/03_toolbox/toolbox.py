####################################################################################################
# LAB 3 — TOOLBOX (bundling multiple tools into one reusable ToolSet)
#
# WHY THIS MATTERS
#   02_tools/ showed one tool at a time wired to one agent. Real agents need several tools at
#   once (a fact lookup AND a calculator, say) without hand-assembling `tools=[...a, ...b, ...c]`
#   everywhere they're used. A toolbox is that assembly done once, then reused.
#
# HOW IT LINKS TO THE REST OF THE CHAPTER
#   This module is imported by main.py in this same folder, which attaches the whole bundle to
#   one agent in a single line (`toolset=build_toolbox()`). It combines the exact two tool types
#   from 02_tools/ (function tool + code interpreter) instead of introducing a third kind.
#
# HOW IT WORKS
#   ToolSet() is a container — `.add(tool)` accepts any Tool instance (FunctionTool,
#   CodeInterpreterTool, FileSearchTool, ...). `build_toolbox()` returns one ToolSet holding both
#   custom hotel-lookup functions and the built-in code interpreter. TOOLBOX_FUNCTIONS is exported
#   separately because enable_auto_function_calls() needs the raw Python callables, not the
#   ToolSet wrapper, to know which functions it's allowed to execute locally.
####################################################################################################
from azure.ai.agents.models import CodeInterpreterTool, FunctionTool, ToolSet


def get_checkout_time(hotel_name: str = "Crystal Hotels") -> str:
    """Get the standard checkout time for a Crystal Hotels property.

    :param hotel_name: Name of the hotel property.
    """
    return f"{hotel_name} standard checkout time is 12:00 PM (late checkout until 2:00 PM on request)."


def get_nightly_rate(room_type: str) -> str:
    """Look up the nightly rate for a Crystal Hotels room type.

    :param room_type: The room type, e.g. "Standard", "Deluxe", "Suite".
    """
    rates = {"standard": 119, "deluxe": 189, "suite": 289}
    rate = rates.get(room_type.lower())
    if rate is None:
        return f"No rate found for room type '{room_type}'."
    return f"{room_type} rooms are ${rate}/night."


def build_toolbox() -> ToolSet:
    """Bundle the hotel function tools and the built-in code interpreter into one reusable ToolSet."""
    toolset = ToolSet()
    toolset.add(FunctionTool(functions={get_checkout_time, get_nightly_rate}))
    toolset.add(CodeInterpreterTool())
    return toolset


TOOLBOX_FUNCTIONS = {get_checkout_time, get_nightly_rate}
