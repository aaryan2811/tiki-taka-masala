"""Read one open DFL / Sportec match (Bassek et al. 2025, CC BY 4.0) into arrays, and cache them.

Each match is three XML files from figshare (article 28196177), kept outside every repository:

    DFL_02_01_matchinformation_<competition>_DFL-MAT-<id>.xml   teams, players, pitch
    DFL_03_02_events_raw_<competition>_DFL-MAT-<id>.xml          hand-coded events
    DFL_04_03_positions_raw_observed_<competition>_DFL-MAT-<id>.xml   25 Hz positions, ~400 MB

The files are downloaded data, so they are only ever parsed, never executed, and the positions file
is streamed with iterparse rather than loaded whole. It holds one FrameSet per person per half,
each a list of <Frame N=... X=... Y=... S=.../>; the ball's FrameSet also carries BallStatus (1
alive, 0 dead) and BallPossession (1 home, 2 away) on every frame. Those two flags are the dev
set's ground truth (truth.py).

What the DFL feasibility audit (a research note kept out of this line) found about the format, and
what this module relies on:
  - frames are numbered consecutively within a half, so the ball's FrameSet is the frame axis and
    a person's frame lands at row N - N0;
  - positions are metres with the origin at the centre spot; S is km/h (kept, in m/s, because M2b's
    slowdown thresholds were calibrated on it: solution/experiments/foul_stoppages.py);
  - the ball is present on every frame, and frozen in place during every dead ball (degrade.py
    replaces that stretch, because it gives the label away).

The parse takes a few seconds per half and is cached as an .npz under solution/devset/cache/, which
is gitignored: it is frame-level tracking, and the dev set never commits data.
"""

import glob
import os
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
CACHE_VERSION = 3  # bump when the arrays below change shape or meaning

SECTIONS = ("firstHalf", "secondHalf")  # league matches: no extra time

# The two matches the audit downloaded first. params.json's matched widths (fit.py) and M2b's
# slowdown thresholds (possession/config.py) were set on them, so their scores are biased; every
# other match is held out and must not be used to set anything.
CALIBRATION = ("J03WMX", "J03WPY")
FPS = 25
KEEPER_POSITION = "TW"  # DFL's PlayingPosition for a goalkeeper (Torwart)
EVENT_SLACK = 2 * FPS  # frames: the paper's event-to-tracking timing noise is about +/-1.8 s
KMH = 3.6  # DFL's S is km/h
BALL, REFEREE = "BALL", "referee"

# Event elements that restart play, as the audit counted them (section 6.4), mapped to the names
# Smart Tagging uses for its "Type" so that tags.py can treat both alike.
SET_PIECE_EVENTS = {
    "KickOff": "Kick Off",
    "ThrowIn": "Throw-in",
    "FreeKick": "Free Kick",
    "GoalKick": "Goal Kick",
    "CornerKick": "Corner Kick",
    "Penalty": "Penalty",
}


class Half(object):
    """One half of a DFL match, every array indexed by row = frame N - n0.

    ball        (n, 3)     x, y, z in metres, origin at the centre spot
    status      (n,)       BallStatus: 1 alive, 0 dead
    possession  (n,)       BallPossession: 1 home, 2 away -- named on every frame, dead balls too
    players     (n, p, 2)  x, y per person; NaN while that person is not on the pitch
    speed       (n, p)     DFL's own speed per person in m/s (its S, km/h, / 3.6); NaN when off
    team        (p,)       0 home, 1 away
    keeper      (p,)       True for a goalkeeper (PlayingPosition TW)
    person_ids  (p,)
    n0, t0      first frame number, and its wall-clock time as a POSIX timestamp
    """

    def __init__(self, section, n0, t0, ball, status, possession, players, team, keeper,
                 person_ids, speed=None):
        self.section = section
        self.n0, self.t0 = int(n0), float(t0)
        self.ball, self.status, self.possession = ball, status, possession
        self.players, self.team, self.keeper = players, team, keeper
        self.speed = speed if speed is not None else np.full(players.shape[:2], np.nan,
                                                             dtype=np.float32)
        self.person_ids = person_ids

    @property
    def n_frames(self):
        return len(self.ball)

    def row_of_time(self, timestamp):
        """Row of a wall-clock POSIX timestamp, assuming 25 frames a second from t0."""
        return int(round((timestamp - self.t0) * FPS))


class Match(object):
    """A parsed DFL match: its halves, set-piece events and metadata."""

    def __init__(self, match_id, halves, set_pieces, team_names, team_ids, pitch):
        self.match_id = match_id
        self.halves = halves              # [Half] in SECTIONS order
        self.set_pieces = set_pieces      # [(half_index, row, type, team or -1)]
        self.team_names = tuple(team_names)
        self.team_ids = tuple(team_ids)   # (home, guest) DFL club ids
        self.pitch = tuple(pitch)


# -- files -----------------------------------------------------------------------------------------

def match_files(dfl_dir, match_id):
    """(info, events, positions) paths for one match id, from <dfl_dir>/<id>/ or <dfl_dir>/."""
    found = {}
    for kind, stem in (("info", "DFL_02_01_matchinformation"), ("events", "DFL_03_02_events_raw"),
                       ("positions", "DFL_04_03_positions_raw_observed")):
        pattern = "{}_*_DFL-MAT-{}.xml".format(stem, match_id)
        hits = (glob.glob(os.path.join(dfl_dir, match_id, pattern))
                or glob.glob(os.path.join(dfl_dir, pattern)))
        if len(hits) != 1:
            raise IOError("{}: expected one {} under {}, found {}".format(
                match_id, pattern, dfl_dir, len(hits)))
        found[kind] = hits[0]
    return found["info"], found["events"], found["positions"]


def available_matches(dfl_dir):
    """Match ids under dfl_dir that have a positions file."""
    paths = glob.glob(os.path.join(dfl_dir, "**", "DFL_04_03_positions_raw_observed_*.xml"),
                      recursive=True)
    return sorted(os.path.basename(path).rsplit("DFL-MAT-", 1)[1][:-len(".xml")] for path in paths)


# -- match information -----------------------------------------------------------------------------

def read_info(info_path):
    """(team_names, team_ids, pitch, keepers) from the match information file.

    team order is (home, guest), the order BallPossession numbers them. keepers is the set of
    PersonIds whose PlayingPosition is TW.
    """
    root = ET.parse(info_path).getroot()
    general = root.find(".//General")
    environment = root.find(".//Environment")
    team_ids = (general.get("HomeTeamId"), general.get("GuestTeamId"))
    team_names = (general.get("HomeTeamName"), general.get("GuestTeamName"))
    pitch = (float(environment.get("PitchX")), float(environment.get("PitchY")))
    keepers = {player.get("PersonId") for player in root.iter("Player")
               if player.get("PlayingPosition") == KEEPER_POSITION}
    return team_names, team_ids, pitch, keepers


# -- positions -------------------------------------------------------------------------------------

def _timestamp(text):
    return datetime.fromisoformat(text).timestamp()


def stream_framesets(positions_path):
    """Yield (section, team_id, person_id, rows) per FrameSet, streamed.

    rows is a list of tuples: (N, X, Y, S) for a person; (N, X, Y, Z, BallStatus, BallPossession)
    for the ball. The first ball row carries its time too, as the frame axis's t0. Every element
    is cleared once read, so memory holds one FrameSet at a time rather than the whole file.
    """
    current, rows, t0 = None, [], None
    for event, element in ET.iterparse(positions_path, events=("start", "end")):
        if event == "start":
            if element.tag == "FrameSet":
                current = (element.get("GameSection"), element.get("TeamId"),
                           element.get("PersonId"))
                rows, t0 = [], None
            continue
        if element.tag == "Frame" and current is not None:
            get = element.get
            if current[1] == BALL:
                if t0 is None:
                    t0 = _timestamp(get("T"))
                rows.append((int(get("N")), float(get("X")), float(get("Y")), float(get("Z")),
                             int(get("BallStatus")), int(get("BallPossession"))))
            elif current[1] != REFEREE:
                rows.append((int(get("N")), float(get("X")), float(get("Y")),
                             float(get("S", "nan"))))
            element.clear()
        elif element.tag == "FrameSet" and current is not None:
            yield current + (rows, t0)
            current, rows = None, []
            element.clear()


def parse_positions(positions_path, team_ids, keepers):
    """[Half] from the positions file. Persons are sorted home first, then by PersonId."""
    sections = {}
    for section, team_id, person_id, rows, t0 in stream_framesets(positions_path):
        entry = sections.setdefault(section, {"ball": None, "t0": None, "persons": []})
        if team_id == BALL:
            entry["ball"], entry["t0"] = rows, t0
        elif team_id in team_ids:
            entry["persons"].append((team_ids.index(team_id), person_id, rows))

    halves = []
    for section in SECTIONS:
        entry = sections.get(section)
        if entry is None or entry["ball"] is None:
            raise ValueError("{}: no ball FrameSet for {}".format(positions_path, section))
        ball_rows = np.array(entry["ball"], dtype=np.float64)
        frames = ball_rows[:, 0].astype(np.int64)
        if not np.all(np.diff(frames) == 1):
            raise ValueError("{}: ball frames of {} are not consecutive".format(positions_path,
                                                                                 section))
        n0, n = int(frames[0]), len(frames)
        persons = sorted(entry["persons"], key=lambda person: (person[0], person[1]))
        players = np.full((n, len(persons), 2), np.nan, dtype=np.float32)
        speed = np.full((n, len(persons)), np.nan, dtype=np.float32)
        for column, (_, _, rows) in enumerate(persons):
            values = np.array(rows, dtype=np.float64)
            index = values[:, 0].astype(np.int64) - n0
            keep = (index >= 0) & (index < n)
            players[index[keep], column] = values[keep, 1:3]
            speed[index[keep], column] = values[keep, 3] / KMH
        halves.append(Half(section, n0, entry["t0"],
                           ball=ball_rows[:, 1:4].astype(np.float32),
                           status=ball_rows[:, 4].astype(np.int8),
                           possession=ball_rows[:, 5].astype(np.int8),
                           players=players,
                           team=np.array([person[0] for person in persons], dtype=np.int8),
                           keeper=np.array([person[1] in keepers for person in persons]),
                           person_ids=np.array([person[1] for person in persons]),
                           speed=speed))
    return halves


# -- events ----------------------------------------------------------------------------------------

def read_set_pieces(events_path, halves, team_ids):
    """[(half_index, row, type, team)] for every set-piece event, sorted.

    row is the event's EventTime on the half's frame axis. The paper puts event timestamps within
    about +/-1.8 s of the tracking, and the audit found set pieces a median -0.5..+0.4 s from the
    end of their dead ball, so a row is only good to a second or two -- which is all a Smart Tagging
    stand-in needs. team is 0 home, 1 guest or -1; a kick-off names its team on the Play inside it.
    """
    out = []
    for event in ET.parse(events_path).getroot().iter("Event"):
        children = list(event)
        if not children or children[0].tag not in SET_PIECE_EVENTS:
            continue
        piece = children[0]
        team_id = piece.get("Team")
        if team_id is None:
            play = piece.find("Play")
            team_id = play.get("Team") if play is not None else None
        timestamp = _timestamp(event.get("EventTime"))
        for index, half in enumerate(halves):
            row = half.row_of_time(timestamp)
            # A half's frames start at its kick-off, and the kick-off event can be stamped a few
            # frames before the first one; within the event-time precision it belongs to the half.
            if -EVENT_SLACK <= row < half.n_frames + EVENT_SLACK:
                team = team_ids.index(team_id) if team_id in team_ids else -1
                out.append((index, min(max(row, 0), half.n_frames - 1),
                            SET_PIECE_EVENTS[piece.tag], team))
                break
    return sorted(out)


# -- cache -----------------------------------------------------------------------------------------

def _stamp(path):
    info = os.stat(path)
    return np.array([CACHE_VERSION, info.st_size, int(info.st_mtime)], dtype=np.int64)


def _save(path, match, stamp):
    store = {"stamp": stamp, "team_names": np.array(match.team_names),
             "team_ids": np.array(match.team_ids), "pitch": np.array(match.pitch),
             "set_pieces": np.array([(h, r, t) for h, r, _, t in match.set_pieces], dtype=np.int64)
             .reshape(-1, 3),
             "set_piece_types": np.array([kind for _, _, kind, _ in match.set_pieces])}
    for index, half in enumerate(match.halves):
        prefix = "h{}_".format(index)
        store.update({prefix + "meta": np.array([half.n0, half.t0]), prefix + "ball": half.ball,
                      prefix + "status": half.status, prefix + "possession": half.possession,
                      prefix + "players": half.players, prefix + "team": half.team,
                      prefix + "keeper": half.keeper, prefix + "person_ids": half.person_ids,
                      prefix + "speed": half.speed})
    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    np.savez(path, **store)


def _restore(match_id, stored):
    halves = []
    for index, section in enumerate(SECTIONS):
        prefix = "h{}_".format(index)
        n0, t0 = stored[prefix + "meta"]
        halves.append(Half(section, n0, t0, stored[prefix + "ball"], stored[prefix + "status"],
                           stored[prefix + "possession"], stored[prefix + "players"],
                           stored[prefix + "team"], stored[prefix + "keeper"],
                           stored[prefix + "person_ids"], stored[prefix + "speed"]))
    set_pieces = [(int(h), int(r), str(kind), int(t)) for (h, r, t), kind in
                  zip(stored["set_pieces"], stored["set_piece_types"])]
    return Match(match_id, halves, set_pieces, [str(n) for n in stored["team_names"]],
                 [str(n) for n in stored["team_ids"]], tuple(float(v) for v in stored["pitch"]))


def load(dfl_dir, match_id, refresh=False, verbose=True):
    """The match as a Match, parsed once and cached; rebuilt when the positions file changes."""
    info_path, events_path, positions_path = match_files(dfl_dir, match_id)
    cache = os.path.join(CACHE_DIR, "{}.npz".format(match_id))
    stamp = _stamp(positions_path)
    if not refresh and os.path.isfile(cache):
        with np.load(cache) as stored:
            if np.array_equal(stored["stamp"], stamp):
                return _restore(match_id, stored)

    if verbose:
        print("parsing {} (cached afterwards) ...".format(os.path.basename(positions_path)))
    team_names, team_ids, pitch, keepers = read_info(info_path)
    halves = parse_positions(positions_path, team_ids, keepers)
    match = Match(match_id, halves, read_set_pieces(events_path, halves, team_ids), team_names,
                  team_ids, pitch)
    _save(cache, match, stamp)
    return match
