"""
The AgentViewer class provides a way to display gridworlds in the terminal.

AgentViewer was created by n0p2.
The original repo can be found at https://github.com/n0p2/ai-safety-gridworlds-viewer

It is used in the gymnasium environment to render the environment for human.
The code is integrated into this repo to simplify dependency management.
"""

import logging
import collections
import curses
import datetime
import time


class AgentViewer:
    """A terminal-based game viewer for ai-safety-gridworlds games.
    (https://github.com/deepmind/ai-safety-gridworlds)

    This is based on the `human_ui.CursesUi` class from the pycolab game
    engine (https://github.com/deepmind/pycolab) developed by Deepmind.
    Both `CursesUi` and its subclass `safety_ui.SafetyCursesUi` allow a
    human player to play their games with keyboard input.

    `AgentViewer` is created to enable display of a live game as an agent
    plays it. This is desirable in reinforcement learning (RL) settings,
    where one needs to view an agent's interactions with the environment
    as the game progresses.

    As far as programming paradigm goes, I try to find a balance
    between OO (object oriented) and FP (functional programming).
    Classes are defined to manage resources and mutable states.
    All other reusable logic are defined as functions (without side
    effect) outside of the class.
    """

    def __init__(self, pause, **kwargs):
        """Construct an `AgentViewer`, which displays agent's interactions
        with the environment in a terminal for ai-safety-gridworlds games
        developed by Google Deepmind.

        Args:
            pause: float.
                A game played by an agent often proceeds at a pace too fast
                for meaningful watching. `pause` allows one to adjust the
                displaying pace. Note that when displaying an elapsed time
                on the game window, the wall clock time consumed by pausing
                is subtracted (see `_get_elapsed`).
        """
        self._initialized = False
        self._screen = None
        self._colour_pair = None
        
        try:
            self._screen = curses.initscr()
            self._colour_pair = init_curses(self._screen, **kwargs)
            self._pause = pause
            self._initialized = True
            self.reset_time()
        except Exception as e:
            # Clean up if initialization fails
            self._safe_cleanup()
            raise RuntimeError(f"Failed to initialize AgentViewer: {e}")

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Silently ignore cleanup errors in destructor
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Safely close the curses window."""
        if self._initialized and self._screen is not None:
            try:
                curses.endwin()
            except Exception:
                # Curses cleanup can fail if terminal is already closed
                pass
            finally:
                self._initialized = False
                self._screen = None

    def _safe_cleanup(self):
        """Emergency cleanup without raising exceptions."""
        try:
            if self._screen is not None:
                curses.endwin()
        except Exception:
            pass
        finally:
            self._initialized = False
            self._screen = None

    def display(self, env):
        """Display the current game state.

        Args:
            env: ai_safety_gridworlds.environments.shared.safety_game.SafetyEnvironment.
                An instance of SafetyEnvironment which contains observations (or boards)
                and returns.
        """
        if not self._initialized:
            raise RuntimeError("AgentViewer not properly initialized")

        board = env.current_game._board.board
        return_ = env.episode_return
        # Time cost is not always a good indicator for performance evaluation.
        # Other indicators, such as number of episodes, might be more suitable.
        # Nevertheless, only elapsed time is displayed, while support of
        # additional information should be done by the consumer of AgentViewer.
        elapsed = self._get_elapsed()
        
        try:
            display(self._screen, board, return_, elapsed, self._colour_pair)
            self._do_pause()
        except Exception as e:
            # On display error, clean up and re-raise
            self._safe_cleanup()
            raise RuntimeError(f"Display failed: {e}")

    def reset_time(self):
        """Reset the internal timer."""
        self._start_time = time.time()
        self._pause_cnt = 0

    def _do_pause(self):
        """Pause for the configured duration."""
        if self._pause is not None:
            time.sleep(self._pause)
            self._pause_cnt += 1

    def _get_elapsed(self):
        """Get elapsed time minus pause time."""
        now = time.time()
        s = 0.0 if self._pause is None else self._pause
        elapsed = now - self._start_time - float(s) * self._pause_cnt
        return elapsed


# --------
# Core functions that deal with screen initialization and display.
# These functions are heavily based on the `human_ui.CursesUi` class
# (https://github.com/deepmind/pycolab/blob/master/pycolab/human_ui.py)
# --------


def display(screen, board, score, elapsed, color_pair):
    """Redraw the game board onto the already-running screen, with elapsed time and score.

    Args:
        screen: Curses screen object
        board: Game board array
        score: Current score
        elapsed: Elapsed time in seconds
        color_pair: Color pair mapping
    """
    screen.clear()

    # Display the game clock and the current score.
    screen.addstr(0, 2, ts2str(elapsed), curses.color_pair(0))
    screen.addstr(0, 10, f"Score: {score:.2f}", curses.color_pair(0))

    # Display game board rows one-by-one.
    for row, board_line in enumerate(board, start=1):
        screen.move(row, 0)  # Move to start of this board row.
        # Display game board characters one-by-one.
        for character in board_line:
            character = int(character)
            color_id = color_pair[chr(character)]
            color_ch = curses.color_pair(color_id)
            screen.addch(character, color_ch)

    screen.refresh()


def init_colour(color_bg, color_fg):
    """Initialize color pairs for curses display.
    
    Based on `human_ui.CursesUi._init_colour`
    (https://github.com/deepmind/pycolab/blob/master/pycolab/human_ui.py)
    """
    curses.start_color()
    # The default color for all characters without colors listed is boring
    # white on black, or "system default", or somesuch.
    colour_pair = collections.defaultdict(lambda: 0)
    
    # And if the terminal doesn't support true color, that's all you get.
    if not curses.can_change_color():
        return colour_pair

    # Collect all unique foreground and background colors. If this terminal
    # doesn't have enough colors for all of the colors the user has supplied,
    # plus the two default colors, plus the largest color id (which we seem
    # not to be able to assign, at least not with xterm-256color) stick with
    # boring old white on black.
    colours = set(color_fg.values()).union(set(color_bg.values()))
    if (curses.COLORS - 2) < len(colours):
        return colour_pair

    # Get all unique characters that have a foreground and/or background color.
    # If this terminal doesn't have enough color pairs for all characters plus
    # the default color pair, stick with boring old white on black.
    characters = set(color_fg).union(color_bg)
    if (curses.COLOR_PAIRS - 1) < len(characters):
        return colour_pair

    # Get the identifiers for both colors in the default color pair.
    cpair_0_fg_id, cpair_0_bg_id = curses.pair_content(0)

    # With all this, make a mapping from colors to the IDs we'll use for them.
    ids = set(range(curses.COLORS - 1)) - {  # The largest ID is not assignable?
        cpair_0_fg_id,
        cpair_0_bg_id,
    }  # We don't want to change these.
    ids = list(reversed(sorted(ids)))  # We use color IDs from large to small.
    ids = ids[: len(colours)]  # But only those color IDs we actually need.
    colour_ids = dict(zip(colours, ids))

    # Program these colors into curses.
    for colour, cid in colour_ids.items():
        curses.init_color(cid, *colour)

    # Now add the default colors to the color-to-ID map.
    cpair_0_fg = curses.color_content(cpair_0_fg_id)
    cpair_0_bg = curses.color_content(cpair_0_bg_id)
    colour_ids[cpair_0_fg] = cpair_0_fg_id
    colour_ids[cpair_0_bg] = cpair_0_bg_id

    # The color pair IDs we'll use for all characters count up from 1; note that
    # the "default" color pair of 0 is already defined, since _colour_pair is a
    # defaultdict.
    colour_pair.update(
        {character: pid for pid, character in enumerate(characters, start=1)}
    )

    # Program these color pairs into curses, and that's all there is to do.
    for character, pid in colour_pair.items():
        # Get foreground and background colors for this character. Note how in
        # the absence of a specified background color, the same color as the
        # foreground is used.
        cpair_fg = color_fg.get(character, cpair_0_fg_id)
        cpair_bg = color_bg.get(character, cpair_fg)
        # Get color IDs for those colors and initialize a color pair.
        cpair_fg_id = colour_ids[cpair_fg]
        cpair_bg_id = colour_ids[cpair_bg]
        curses.init_pair(pid, cpair_fg_id, cpair_bg_id)

    return colour_pair


def init_curses(screen, color_bg, color_fg, delay=None):
    """Initialize curses with color support."""
    logger = get_logger()
    logger.info("init_curses...")
    
    # If the terminal supports color, program the colors into curses as
    # "color pairs". Update our dict mapping characters to color pairs.
    colour_pair = init_colour(color_bg, color_fg)
    curses.curs_set(0)  # We don't need to see the cursor.
    
    if delay is None:
        screen.timeout(-1)  # Blocking reads
    else:
        screen.timeout(delay)  # Nonblocking (if 0) or timing-out reads

    logger.info("init_curses success.")
    return colour_pair


def ts2str(ts_delta):
    """Convert timestamp delta to string format."""
    delta = datetime.timedelta(seconds=ts_delta)
    return str(delta).split(".")[0]


# --------
# logging and debugging
# --------
_logger = None


def get_logger():
    """Get or create singleton logger instance."""
    global _logger
    if _logger is None:
        _logger = logging.getLogger(__name__)
        hdlr = logging.FileHandler(__name__ + ".log")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        hdlr.setFormatter(formatter)
        _logger.addHandler(hdlr)
        _logger.setLevel(logging.DEBUG)

    return _logger
