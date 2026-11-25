"""Reads config.ini and serves its values to the other scripts.

Every script uses these as defaults. A command-line argument always wins, so a
blank entry in config.ini is only a problem if nothing supplies the value.

    from config import settings

    settings.path("project", "folder")      # Path, or None if blank
    settings.text("grid", "resample")       # str, or None
    settings.number("grid", "cell_size_m")  # float, or None
    settings.flag("rainfall", "drop_first_step")
    settings.require("project", "folder")   # Path, or raises

config.ini need not exist: every value can be given on the command line
instead. Check settings.exists when a script cannot run without the file.
"""

from __future__ import annotations

import configparser
from pathlib import Path

__all__ = ["ConfigError", "Settings", "settings",
           "CONFIG_FILE", "TEMPLATE_FILE"]

CONFIG_FILE = Path(__file__).with_name("config.ini")
TEMPLATE_FILE = Path(__file__).with_name("config.example.ini")

_TRUE = {"yes", "true", "1", "on"}
_FALSE = {"no", "false", "0", "off", ""}


class ConfigError(RuntimeError):
    """A value is missing or cannot be read."""


class Settings:
    """Values from config.ini. Blank entries read as None."""

    def __init__(self, file: Path = CONFIG_FILE) -> None:
        self.file = file
        self._parser = configparser.ConfigParser(inline_comment_prefixes=("#",))
        if file.exists():
            self._parser.read(file, encoding="utf-8")

    @property
    def exists(self) -> bool:
        """Whether config.ini is actually present."""
        return self.file.exists()

    # -- raw ---------------------------------------------------------------

    def text(self, section: str, key: str) -> str | None:
        """String value, or None if blank or absent."""
        if not self._parser.has_option(section, key):
            return None
        value = self._parser.get(section, key).strip()
        return value or None

    # -- typed -------------------------------------------------------------

    def path(self, section: str, key: str) -> Path | None:
        """Path value. Relative paths resolve against this file's folder."""
        value = self.text(section, key)
        if value is None:
            return None
        candidate = Path(value.strip('"').strip("'"))
        return candidate if candidate.is_absolute() else (self.file.parent / candidate)

    def number(self, section: str, key: str) -> float | None:
        value = self.text(section, key)
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            raise ConfigError(
                "[%s] %s = %r in %s is not a number."
                % (section, key, value, self.file.name)
            ) from None

    def integer(self, section: str, key: str) -> int | None:
        value = self.number(section, key)
        return None if value is None else int(value)

    def flag(self, section: str, key: str) -> bool:
        value = (self.text(section, key) or "").lower()
        if value in _TRUE:
            return True
        if value in _FALSE:
            return False
        raise ConfigError(
            "[%s] %s = %r in %s is not yes or no."
            % (section, key, value, self.file.name)
        )

    # -- enforcement -------------------------------------------------------

    def require(self, section: str, key: str) -> Path:
        """Path value that must be present."""
        value = self.path(section, key)
        if value is None:
            raise ConfigError(
                "[%s] %s is blank in %s, and was not given on the command line."
                % (section, key, self.file.name)
            )
        return value

    def missing(self, *wanted: tuple[str, str]) -> list[str]:
        """Which of the given (section, key) pairs are blank."""
        return ["[%s] %s" % (s, k) for s, k in wanted if self.text(s, k) is None]

    def report(self) -> str:
        """One line per setting, for the start of a run."""
        lines = []
        for section in self._parser.sections():
            for key, value in self._parser.items(section):
                lines.append("    %-28s %s"
                             % ("%s.%s" % (section, key), value.strip() or "(blank)"))
        return "\n".join(lines)


settings = Settings()
