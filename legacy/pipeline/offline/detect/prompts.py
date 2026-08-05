"""Vocabulary and prompt data for the hybrid offline detector.

Pure data module — no model dependencies, no GPU required.

YOLO-World vocabs
-----------------
Five fine-grained vocabs are merged into two *super-vocabs* to halve the
number of ``set_classes()`` text-encoder re-runs per batch:

  BROAD_VOCAB            = GENERIC_OBJECTS + PUBLIC_INFRA  (always-on pass)
  <venue>_SPECIFIC_VOCAB = domain-specific categories       (second pass)

VENUE_YOLO_VOCABS maps each venue name to the right specific vocab.

GroundingDINO prompts
---------------------
VENUE_PROMPTS maps each venue name to a dot-separated GDINO prompt focused
on multi-word, descriptive phrases that YOLO's open-vocab struggles with
(e.g. "automatic border control gate", "out of gauge baggage scanner").

Adding a new venue
------------------
1. Define a raw vocab tuple (e.g. RETAIL_VOCAB) with category strings.
2. Build a merged super-vocab using _dedupe_normalized().
3. Add the venue to VENUE_YOLO_VOCABS.
4. Add a GDINO prompt string to VENUE_PROMPTS.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_yolo_category(name: str) -> str:
    return " ".join(name.lower().strip().split())


def _dedupe_normalized(*vocabs: Tuple[str, ...]) -> Tuple[str, ...]:
    """Merge vocabs, normalise each category, dedupe preserving
    first-occurrence order."""
    seen: Dict[str, None] = {}
    for vocab in vocabs:
        for cat in vocab:
            seen.setdefault(_normalize_yolo_category(cat), None)
    return tuple(seen.keys())


def _build_yolo_vocabulary_passes(
    *,
    enable_broad: bool,
    resolved_vocab: Optional[Tuple[str, ...]],
    conf_broad: float,
    conf_specific: float,
) -> List[Tuple[Tuple[str, ...], float]]:
    """Build the ordered list of (vocab, confidence) pairs for YOLO inference."""
    passes: List[Tuple[Tuple[str, ...], float]] = []
    if enable_broad:
        passes.append((BROAD_VOCAB, float(conf_broad)))
    if resolved_vocab:
        passes.append((resolved_vocab, float(conf_specific)))
    return passes


# ---------------------------------------------------------------------------
# People-class drop list (applied after every detection pass)
# ---------------------------------------------------------------------------

DROP_LABELS_LOWER = frozenset(
    {
        "person",
        "people",
        "human",
        "pedestrian",
        "passenger",
        "man",
        "woman",
        "child",
        "crowd",
    }
)


# ---------------------------------------------------------------------------
# Raw YOLO-World vocabularies
# ---------------------------------------------------------------------------

GENERIC_OBJECTS_VOCAB: Tuple[str, ...] = (
    "machine",
    "cabinet",
    "rack",
    "cart",
    "container",
    "tool cabinet",
    "storage cabinet",
    "crate",
    "box",
    "shelf",
    "industrial equipment",
    "electrical equipment",
    "mechanical equipment",
)

PUBLIC_INFRA_VOCAB: Tuple[str, ...] = (
    "door",
    "automatic door",
    "window",
    "stairs",
    "escalator",
    "elevator",
    "railing",
    "column",
    "pillar",
    "bench",
    "chair",
    "desk",
    "table",
    "trash bin",
    "plant",
    "light",
    "ceiling light",
    "lamp",
    "screen",
    "monitor",
    "display",
    "sign",
    "directory board",
    "barrier",
    "queue barrier",
    "fence",
)

AIRPORT_OPERATIONS_VOCAB: Tuple[str, ...] = (
    "check in counter",
    "baggage drop counter",
    "check in kiosk",
    "self check in kiosk",
    "kiosk",
    "e gate",
    "boarding pass gate",
    "passport control booth",
    "x ray machine",
    "security scanner",
    "metal detector",
    "baggage carousel",
    "conveyor",
    "luggage cart",
    "flight information display",
    "departure board",
    "arrival board",
    "information screen",
)

TRAIN_STATION_OPERATIONS_VOCAB: Tuple[str, ...] = (
    "ticket machine",
    "ticket validator",
    "ticket scanner",
    "fare gate",
    "ticket gate",
    "turnstile",
    "card reader",
    "ticket office",
    "ticket counter",
    "departure board",
    "arrival board",
    "timetable display",
    "timetable screen",
    "train information display",
    "platform sign",
    "platform number sign",
    "platform letter sign",
    "track number sign",
    "train station sign",
    "wayfinding sign",
    "direction sign",
    "line map",
    "station map",
    "map board",
    "clock",
    "help point",
    "emergency phone",
    "defibrillator",
    "fire hose cabinet",
    "platform screen door",
)

TRAIN_STATION_PUBLIC_INFRA_VOCAB: Tuple[str, ...] = (
    "bench",
    "platform bench",
    "waiting bench",
    "seat",
    "waiting area seat",
    "chair",
    "trash bin",
    "recycling bin",
    "vending machine",
    "ATM",
    "luggage locker",
    "information kiosk",
    "advertisement panel",
    "poster",
    "digital display",
    "ceiling light",
    "lamp",
    "speaker",
    "intercom",
    "elevator button panel",
    "handrail",
    "railing",
    "queue barrier",
    "potted plant",
    "vase",
)

SAFETY_SECURITY_VOCAB: Tuple[str, ...] = (
    "CCTV camera",
    "security camera",
    "surveillance camera",
    "fire standpipe",
    "fire extinguisher",
    "smoke detector",
    "sprinkler",
    "fire alarm panel",
    "emergency exit sign",
    "alarm",
    "speaker",
    "intercom",
    "emergency light",
)

TECHNICAL_MEP_VOCAB: Tuple[str, ...] = (
    "generator",
    "generator set",
    "diesel generator",
    "electrical panel",
    "electrical cabinet",
    "switchgear",
    "transformer",
    "power cabinet",
    "control panel",
    "server rack",
    "chiller",
    "HVAC unit",
    "air conditioning unit",
    "cooling tower",
    "boiler",
    "pump",
    "pipe",
    "ventilation duct",
    "industrial machine",
    "motor",
)

HOTEL_FURNITURE_VOCAB: Tuple[str, ...] = (
    # Bedroom furniture & bedding
    "bed",
    "sofa",
    "armchair",
    "wardrobe",
    "bedside table",
    "dressing table",
    # Hospitality appliances / electronics
    "television",
    "minibar",
    "refrigerator",
    "coffee machine",
    "kettle",
    "microwave",
    # Bathroom fixtures
    "bathtub",
    "shower",
    "toilet",
    "sink",
    "mirror",
    "hair dryer",
    # Hotel service & housekeeping equipment
    "cleaning cart",
    "laundry cart",
    "linen cart",
    "housekeeping trolley",
    "luggage rack",
    # Reception & access control
    "keycard reader",
    "reception desk",
    "concierge desk",
    "room safe",
    # Lighting — "light", "lamp", "ceiling light" are in PUBLIC_INFRA_VOCAB (BROAD);
    # "wall lamp" is hotel-specific and not covered by any other YOLO pass.
    "wall lamp",
    # Other
    "vending machine",
    "decorative plant",
    "potted plant",
)

# Shared-space categories: restaurant, bar, banquet hall, lobby, spa.
# Note: "chair" and "table" already live in PUBLIC_INFRA_VOCAB (BROAD); these
# entries add the hospitality-specific variants and fixtures that YOLO won't
# match against the generic labels.
HOTEL_SHARED_SPACES_VOCAB: Tuple[str, ...] = (
    # Dining & bar seating
    "bar stool",
    "dining chair",
    "lounge chair",
    "booth seat",
    "chair",
    "stool",
    "seat",
    # Surfaces & counters
    "bar counter",
    "buffet counter",
    "dining table",
    "coffee table",
    "side table",
    "bar table",
    "reception counter",
    "table",
    # F&B service equipment
    "serving cart",
    "room service trolley",
    "wine rack",
    "tray stand",
    "wine cooler",
    "ice bucket",
    "cutlery tray",
    # Lighting fixtures not in PUBLIC_INFRA (supplements "lamp"/"ceiling light")
    "pendant light",
    "chandelier",
    "floor lamp",
    # Decorative & lobby items
    "decorative vase",
    "flower arrangement",
    "painting",
    "artwork",
    "decorative mirror",
    "coat rack",
    "umbrella stand",
    # Lobby / concierge equipment
    "bellhop cart",
    "luggage trolley",
    "display case",
    # Conference / banquet
    "folding chair",
    "banquet chair",
    "podium",
    "lectern",
    "projector screen",
    "whiteboard",
    "flip chart",
)


# ---------------------------------------------------------------------------
# Merged YOLO super-vocabs (2 forward calls per batch instead of 5+)
# ---------------------------------------------------------------------------

# First pass — always active regardless of venue.
BROAD_VOCAB: Tuple[str, ...] = _dedupe_normalized(
    GENERIC_OBJECTS_VOCAB, PUBLIC_INFRA_VOCAB
)

# Second pass — venue-specific operational categories.
AIRPORT_SPECIFIC_VOCAB: Tuple[str, ...] = _dedupe_normalized(
    AIRPORT_OPERATIONS_VOCAB,
    SAFETY_SECURITY_VOCAB,
    TECHNICAL_MEP_VOCAB,
)
TRAIN_STATION_SPECIFIC_VOCAB: Tuple[str, ...] = _dedupe_normalized(
    TRAIN_STATION_OPERATIONS_VOCAB,
    TRAIN_STATION_PUBLIC_INFRA_VOCAB,
    SAFETY_SECURITY_VOCAB,
)
HOTEL_SPECIFIC_VOCAB: Tuple[str, ...] = _dedupe_normalized(
    HOTEL_FURNITURE_VOCAB,
    HOTEL_SHARED_SPACES_VOCAB,
    SAFETY_SECURITY_VOCAB,
)

# ---------------------------------------------------------------------------
# Venue → YOLO specific vocab mapping
#
# Keys must match the venue names in VENUE_PROMPTS below.
# ---------------------------------------------------------------------------

VENUE_YOLO_VOCABS: Dict[str, Tuple[str, ...]] = {
    "airport": AIRPORT_SPECIFIC_VOCAB,
    "train_station": TRAIN_STATION_SPECIFIC_VOCAB,
    "commercial_center": BROAD_VOCAB,  # broad already covers most retail items
    "factory": _dedupe_normalized(TECHNICAL_MEP_VOCAB, SAFETY_SECURITY_VOCAB),
    "hotel": HOTEL_SPECIFIC_VOCAB,
}

# ---------------------------------------------------------------------------
# GroundingDINO venue prompts
#
# Focus on multi-word / descriptive phrases YOLO's open-vocab struggles with.
# Keep each prompt well under max_length=256 tokens.
# ---------------------------------------------------------------------------

VENUE_PROMPTS: Dict[str, str] = {
    "airport": (
        # Critical overlap with YOLO (kept for safety/ops second opinion)
        "CCTV camera . security camera . surveillance camera . "
        "x ray machine . security scanner . "
        "departure board . arrival board . flight information display . "
        "e gate . boarding pass gate . "
        "self check in kiosk . check in kiosk . "
        # GDINO-niche multi-word phrases YOLO struggles with
        "access control gate . automatic border control gate . "
        "airline check in counter . check in desk . airport kiosk . "
        "baggage x ray machine . carry on baggage scanner . "
        "oversized baggage scanner . out of gauge baggage scanner . "
        "emergency generator . backup generator . emergency power plant . "
        "heating plant . cooling plant . "
        "FIDS screen . information display screen . "
        "passenger boarding bridge . jetway ."
    ),
    "train_station": (
        # Safety/ops second opinion; same high-value items are in the YOLO pass.
        "security camera . CCTV camera . fire extinguisher . fire alarm . "
        "fire hose cabinet . defibrillator . help point . emergency phone . "
        "emergency exit sign . fare gate . ticket gate . ticket validator . "
        # GDINO-niche station phrases and signage YOLO-World can miss.
        "platform number sign . platform letter sign . track number sign . "
        "train station sign . wayfinding sign . platform direction sign . "
        "transfer sign . tactile floor . tactile track . tactile paving . "
        "tactile guidance strip . timetable screen . train information display . "
        "service disruption board . elevator button panel . evacuation map . "
        "carriage number sign ."
    ),
    "commercial_center": (
        "shop sign . store sign . brand logo . price tag . "
        "product shelf . product display . sales counter . "
        "checkout counter . cash register . shopping cart . "
        "mannequin . fitting room sign . restroom sign . "
        "elevator floor indicator . escalator sign . food court sign . "
        "directory board ."
    ),
    "factory": (
        "industrial machine . conveyor belt . control panel . "
        "electrical cabinet . workbench . tool cabinet . tool cart . "
        "pallet . crate . forklift . ladder . hose reel . "
        "safety helmet station . emergency stop button . warning sign . "
        "production line . assembly station . robotic arm ."
    ),
    # Hotel GDINO prompt structure follows the airport pattern:
    #   • Critical-overlap section  — also in YOLO, kept for second-opinion recall
    #   • GDINO-niche section       — multi-word/specific phrases YOLO misses
    #
    # Critical overlaps (intentional — high-value categories where missing one
    # detector's result would be unacceptable):
    #   Security:         already in YOLO's SAFETY_SECURITY_VOCAB
    #   Bedroom furniture: already in YOLO's HOTEL_FURNITURE_VOCAB
    "hotel": (
        # --- Security (second opinion; same items are in HOTEL_SPECIFIC_VOCAB) ---
        "security camera . CCTV camera . surveillance camera . "
        "fire extinguisher . smoke detector . sprinkler . "
        "emergency exit sign . fire alarm . emergency light . intercom . "
        # --- Bedroom furniture (second opinion; in HOTEL_FURNITURE_VOCAB) ---
        "bed . sofa . armchair . wardrobe . bedside table . desk . dresser ."
        # --- Shared spaces (second opinion; in HOTEL_SHARED_SPACES_VOCAB) ---
        "dining chair . dining table . round table . rectangular table . "
        # --- GDINO-niche: bedding details YOLO's 'bed' label won't capture ---
        "pillow . blanket . duvet . bedsheet . mattress . towel . "
        # --- GDINO-niche: room items with precise or multi-word descriptors ---
        "wall lamp . remote control . telephone . clothes hanger . luggage cart . "
        # --- GDINO-niche: signage (multi-word, language-grounding advantage) ---
        "digital display . information sign . direction sign . exit sign . "
        # --- GDINO-niche: service & office equipment ---
        "handrail . vacuum cleaner . computer . keyboard . printer . "
        "cash register . office chair . interior plant"
    ),
}

DEFAULT_VENUE = "train_station"
DEFAULT_GDINO_LABEL = "gdino_venue"
