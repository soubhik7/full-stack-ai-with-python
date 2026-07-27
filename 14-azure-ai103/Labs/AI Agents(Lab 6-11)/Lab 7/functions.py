# =============================================================================
# LAB 7 (helper module): plain Python "tool" functions the astronomy agent
# in agent.py can call. These are NOT agent-specific code - they're ordinary
# functions that happen to get wired up as agent tools in agent.py.
#
# This file needs a `data/` folder next to it (not included in this repo)
# with three pipe-delimited ("|") text files:
#   data/events.txt              - one line per event: Name|Type|MM-DD|loc1;loc2;loc3
#   data/telescope_rates.txt     - one line per tier:  tier_name|hourly_rate
#   data/priority_multipliers.txt- one line per level:  priority_name|multiplier
#
# Example data/events.txt line:
#   Total Solar Eclipse|Eclipse|04-08|north_america;south_america
# Example data/telescope_rates.txt line:
#   standard|50.0
# =============================================================================

import json                        # Converts Python dicts to JSON strings (tool outputs must be strings)
from datetime import datetime      # Used to compare today's date against event dates


def _load_events(file_path: str = "data/events.txt") -> list:
    # Leading underscore = "private helper", not meant to be called directly
    # from outside this module.
    events = []
    with open(file_path) as f:
        for line in f:
            # Split each line on "|" into its 4 fields: name, type, date, locations
            parts = line.strip().split("|")
            if len(parts) == 4:
                # parts[2] looks like "04-08" -> split into month=4, day=8
                month, day = map(int, parts[2].split("-"))
                events.append((
                    parts[0],                          # name
                    parts[1],                          # type
                    month * 100 + day,                 # sortable month-day int, e.g. April 8 -> 408 (lets us compare/sort dates without a full datetime)
                    parts[2],                          # month-day string, e.g. "04-08" (kept for display)
                    set(parts[3].split(";")),          # locations as a set, e.g. {"north_america", "south_america"} - a set makes "is this location in here?" checks fast and ignores duplicate order
                ))
    # Sort all events by their sortable month-day integer, earliest first,
    # so the search loop in next_visible_event() can simply scan forward.
    events.sort(key=lambda e: e[2])
    return events


def _load_rates(file_path: str) -> dict:
    # Generic loader shared by both the telescope-rate file and the
    # priority-multiplier file - both are just "key|number" per line.
    rates = {}
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 2:
                rates[parts[0]] = float(parts[1])
    return rates

# These three module-level constants are loaded ONCE, the moment this file is
# first imported (not on every function call) - so the data files must exist
# before agent.py's `from functions import ...` line succeeds.
EVENTS = _load_events()
TELESCOPE_RATES = _load_rates("data/telescope_rates.txt")
PRIORITY_MULTIPLIERS = _load_rates("data/priority_multipliers.txt")

# Determine the next visible astronomical event for a given location
# Determine the next visible astronomical event for a given location
def next_visible_event(location: str) -> str:
    """Returns the next visible astronomical event for a location."""
    # This is one of the three functions exposed to the agent as a
    # "FunctionTool" in agent.py - the agent decides WHEN to call it and
    # supplies the `location` argument based on the user's question.

    # Build today's date as an MMDD integer (e.g. July 27 -> 727) so it can
    # be compared directly against the same "date" integer stored in EVENTS.
    today = int(datetime.now().strftime("%m%d"))
    # Normalize the location the model gives us (e.g. "North America" ->
    # "north_america") to match the lowercase/underscore format used in the
    # data file.
    loc = location.lower().replace(" ", "_")

    # Retrieve the next event visible from the location, starting with events later this year
    # EVENTS is already sorted earliest-to-latest, so the first matching
    # event we find (that's in this location AND on/after today) is
    # guaranteed to be the *next* one.
    for name, event_type, date, date_str, locs in EVENTS:
        if loc in locs and date >= today:
            # Every tool function must return a STRING (usually JSON) - the
            # agent framework can't accept a raw Python dict back from a
            # function call.
            return json.dumps({"event": name, "type": event_type, "date": date_str, "visible_from": sorted(locs)})

    # No matching future event was found for this location this year.
    return json.dumps({"message": f"No upcoming events found for {location}."})

# Calculate the cost of telescope observation time based on the tier, hours, and priority
def calculate_observation_cost(telescope_tier: str, hours: float, priority: str) -> str:
    """Calculates the cost of telescope observation time."""
    # Normalize case so "Standard"/"standard"/"STANDARD" all match the keys
    # loaded from the data files.
    tier = telescope_tier.lower()
    pri = priority.lower()

    # Validate the tier and priority against what's actually in the data
    # files, and validate hours is positive - returning a clear JSON error
    # message (rather than raising an exception) so the agent can read the
    # problem and explain it to the user in natural language.
    if tier not in TELESCOPE_RATES:
        return json.dumps({"error": f"Unknown telescope tier '{telescope_tier}'. Choose from: {', '.join(TELESCOPE_RATES)}"})

    if pri not in PRIORITY_MULTIPLIERS:
        return json.dumps({"error": f"Unknown priority '{priority}'. Choose from: {', '.join(PRIORITY_MULTIPLIERS)}"})

    if hours <= 0:
        return json.dumps({"error": "Hours must be greater than zero."})

    # Simple pricing formula: hourly rate * hours * priority multiplier
    # (e.g. a "high" priority booking might cost 1.5x the base rate).
    base_cost = TELESCOPE_RATES[tier] * hours
    multiplier = PRIORITY_MULTIPLIERS[pri]
    total_cost = base_cost * multiplier

    # Return every intermediate number too (not just the total) so the
    # agent has enough detail to explain the breakdown if the user asks.
    return json.dumps({
        "telescope_tier": tier,
        "hours": hours,
        "hourly_rate": TELESCOPE_RATES[tier],
        "priority": pri,
        "priority_multiplier": multiplier,
        "base_cost": base_cost,
        "total_cost": total_cost
    })

# Generate an observation report summarizing the details of an astronomical observation session
def generate_observation_report(event_name: str, location: str, telescope_tier: str, hours: float, priority: str, observer_name: str) -> str:
    """
    Generates an observation session report and saves it to a file.

    Returns:
        JSON string with the file path of the generated report.
    """
    # This third tool DEMONSTRATES CALLING THE OTHER TWO TOOLS internally -
    # it's not just a thin wrapper, it reuses calculate_observation_cost and
    # next_visible_event to assemble a combined report, then writes that
    # report out to a local .txt file (a side effect the agent can't see
    # directly, only the confirmation message returned below).
    cost_result = json.loads(calculate_observation_cost(telescope_tier, hours, priority))
    event_result = json.loads(next_visible_event(location))

    # If the cost calculation failed (bad tier/priority/hours), bail out
    # early and surface that same error instead of writing a broken report.
    if "error" in cost_result:
        return json.dumps(cost_result)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    # Build a safe, readable filename from the event name and timestamp,
    # e.g. "report_total_solar_eclipse_2026-07-27_1430.txt".
    filename = f"report_{event_name.replace(' ', '_').lower()}_{timestamp.replace(':', '').replace(' ', '_')}.txt"

    # A plain formatted text report - this is what actually gets saved to
    # disk. Everything here is just an f-string template filling in the
    # numbers computed above.
    report = f"""======================================
  CONTOSO OBSERVATORIES - SESSION REPORT
======================================
Date:           {timestamp}
Observer:       {observer_name}
Event:          {event_name}
Location:       {location}

NEXT VISIBLE EVENT
  Event:        {event_result.get('event', 'N/A')}
  Date:         {event_result.get('date', 'N/A')}

TELESCOPE BOOKING
  Tier:         {cost_result['telescope_tier']}
  Hours:        {cost_result['hours']}
  Hourly Rate:  ${cost_result['hourly_rate']:.2f}
  Priority:     {cost_result['priority']}
  Multiplier:   {cost_result['priority_multiplier']}x

COST SUMMARY
  Base Cost:    ${cost_result['base_cost']:.2f}
  Total Cost:   ${cost_result['total_cost']:.2f}
======================================
"""

    # Write the report to the current working directory (wherever you ran
    # `python agent.py` from) - not inside data/.
    with open(filename, "w") as f:
        f.write(report)

    # The agent only gets told the filename, not the report contents - it
    # will typically just confirm to the user that the report was saved.
    return json.dumps({"status": "Report generated", "file": filename})
