"""Stage 0: parse the raw ATD once into arrays, and cache them.

The raw file is one line per video frame, "<frame_id>:<slot>;<slot>;...:<ball>", each group
comma-separated fields starting "x,y,speed", with NaN for a missing observation. starter/baseline.py
streams it for one answer; every later stage here needs the whole thing, so it is parsed into arrays
once and kept as an .npz under solution/cache/ (gitignored -- it is frame-level tracking).

Positions come back in metres, origin at a pitch corner, because every threshold in config.py is in
metres. Row i is video frame i (frame_id - 1), so arrays index the same way a submission does.

Stage 0 also asks what the ball group carries beyond x and y. All of its fields are kept in
ball_fields so that question can be answered from the cache: a team or ball-state channel would
change the plan (DESIGN.md, stage 0).
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

from . import repo
from .repo import baseline, games

CACHE_DIR = os.path.join(repo.SOLUTION_DIR, "cache")
CACHE_VERSION = 1  # bump when the arrays below change shape or meaning

FIELD_SEP = ":"
GROUP_SEP = ";"
VALUE_SEP = ","


class Tracking(object):
    """The parsed ATD for one recording.

    ball          (n, 2)        ball x, y in metres; NaN where not tracked
    ball_fields   (n, k)        every field of the ball group, as delivered (normalised, raw)
    players       (n, s, 2)     x, y in metres per track slot; NaN where not tracked
    speed         (n, s)        speed in m/s per track slot, as delivered
    team_of_slot  (s,)          0 home, 1 away -- a slot is a tracklet, not a person
    team_names    (home, away)
    pitch         (length, width) in metres
    keeper_slots  slots the XML flags as goalkeepers -- one per team, but see keepers.py
    home_attacks_right_first_half
                  from the XML's attack_direction_first_half; the home team defends the other end
    """

    def __init__(self, ball, ball_fields, players, speed, team_of_slot, team_names, pitch,
                 keeper_slots=(), home_attacks_right_first_half=True):
        self.ball = ball
        self.ball_fields = ball_fields
        self.players = players
        self.speed = speed
        self.team_of_slot = team_of_slot
        self.team_names = tuple(team_names)
        self.pitch = tuple(pitch)
        self.keeper_slots = tuple(keeper_slots)
        self.home_attacks_right_first_half = bool(home_attacks_right_first_half)

    @property
    def n_frames(self):
        return len(self.ball)


def _fields(group, width):
    """One group's values. A missing group may arrive as a bare 'NaN' rather than k of them."""
    if group[:1] == "N" and VALUE_SEP not in group:
        return [np.nan] * width
    return group.split(VALUE_SEP)


def _lines(raw_path):
    """(row, slot_text, ball_text) per well-formed line; row is the video frame."""
    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(FIELD_SEP)
            if len(parts) == 3:
                yield int(parts[0]) - 1, parts[1], parts[2]  # frame_id counts from 1


def group_widths(raw_path):
    """(k_ball, k_slot): fields per ball group and per slot group, each from its first present group.

    Read separately because no line has every group present: there are 46 slots for 22 players, so
    on every frame about half the slots are empty.
    """
    k_ball = k_slot = None
    for _, slots, ball in _lines(raw_path):
        if k_ball is None and ball[:1] != "N":
            k_ball = len(ball.split(VALUE_SEP))
        if k_slot is None:
            present = next((group for group in slots.split(GROUP_SEP) if group[:1] != "N"), None)
            if present is not None:
                k_slot = len(present.split(VALUE_SEP))
        if k_ball is not None and k_slot is not None:
            return k_ball, k_slot
    raise ValueError("{}: never saw both a ball group and a slot group present".format(raw_path))


def parse_raw(raw_path, n_frames, n_slots):
    """(ball_fields, slot_fields) in normalised units, straight off the raw file.

    ball_fields is (n_frames, k_ball) and slot_fields (n_frames, n_slots, k_slot). Frames absent
    from the file stay NaN.
    """
    k_ball, k_slot = group_widths(raw_path)
    ball_fields = np.full((n_frames, k_ball), np.nan, dtype=np.float32)
    slot_fields = np.full((n_frames, n_slots, k_slot), np.nan, dtype=np.float32)

    for row, slots, ball in _lines(raw_path):
        ball_fields[row] = _fields(ball, k_ball)
        values = slots.replace(GROUP_SEP, VALUE_SEP).split(VALUE_SEP)
        if len(values) == n_slots * k_slot:
            slot_fields[row] = np.array(values, dtype=np.float32).reshape(n_slots, k_slot)
            continue
        # Some group is a bare NaN, or the line disagrees with the metadata: the slow path says which.
        groups = slots.split(GROUP_SEP)
        if len(groups) != n_slots:
            raise ValueError("{}: frame_id {} has {} slot groups, metadata says {}".format(
                raw_path, row + 1, len(groups), n_slots))
        slot_fields[row] = [_fields(group, k_slot) for group in groups]
    return ball_fields, slot_fields


def _parameter(element, name):
    """The Value of the ProviderParameter called name under element, or None."""
    for parameter in element.iter("ProviderParameter"):
        if parameter.findtext("Name") == name:
            return parameter.findtext("Value")
    return None


def keeper_slots(meta_path):
    """Slots whose position_type is Goalkeeper, in the XML's Player order (the slot order)."""
    root = ET.parse(meta_path).getroot()
    return tuple(index for index, player in enumerate(root.iter("Player"))
                 if _parameter(player, "position_type") == "Goalkeeper")


def home_attacks_right_first_half(meta_path):
    """True if the first team in the file -- the home side -- attacks left to right in the first
    half. Positions run from the pitch corner at x 0, so that team defends the low-x goal."""
    root = ET.parse(meta_path).getroot()
    home = next(root.iter("Team"))
    return _parameter(home, "attack_direction_first_half") == "left_to_right"


def from_fields(ball_fields, slot_fields, team_of_slot, team_names, pitch, keepers=(),
                home_attacks_right=True):
    """Tracking in metres from normalised fields."""
    scale = np.array(pitch, dtype=np.float32)
    return Tracking(ball=ball_fields[:, :2] * scale,
                    ball_fields=ball_fields,
                    players=slot_fields[:, :, :2] * scale,
                    speed=slot_fields[:, :, 2],
                    team_of_slot=np.asarray(team_of_slot, dtype=np.int8),
                    team_names=team_names,
                    pitch=pitch,
                    keeper_slots=keepers,
                    home_attacks_right_first_half=home_attacks_right)


def _cache_path(game, source):
    return os.path.join(CACHE_DIR, "{}_{}_tracking.npz".format(game, source))


def _stamp(path):
    """Identifies the exact raw file a cache was built from."""
    info = os.stat(path)
    return np.array([CACHE_VERSION, info.st_size, int(info.st_mtime)], dtype=np.int64)


def load(game=games.GAME, data_dir=None, source="tactical", refresh=False, verbose=True):
    """The ATD of one recording as a Tracking, parsed once and cached.

    The cache is rebuilt whenever the raw file's size or modification time changes, or with
    refresh=True.
    """
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game, data_dir, source)
    team_of_slot = baseline.team_of_each_track(meta_path)
    team_names = baseline.team_names(meta_path)
    pitch = games.GAMES[game]["pitch"]
    keepers = keeper_slots(meta_path)
    home_attacks_right = home_attacks_right_first_half(meta_path)

    cache = _cache_path(game, source)
    stamp = _stamp(raw_path)
    if not refresh and os.path.isfile(cache):
        with np.load(cache) as stored:
            if np.array_equal(stored["stamp"], stamp):
                return from_fields(stored["ball_fields"], stored["slot_fields"], team_of_slot,
                                   team_names, pitch, keepers, home_attacks_right)

    if verbose:
        print("parsing {} (cached afterwards) ...".format(os.path.basename(raw_path)))
    ball_fields, slot_fields = parse_raw(raw_path, games.source_frames(source, game),
                                         len(team_of_slot))
    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    np.savez(cache, ball_fields=ball_fields, slot_fields=slot_fields, stamp=stamp)
    return from_fields(ball_fields, slot_fields, team_of_slot, team_names, pitch, keepers,
                       home_attacks_right)
