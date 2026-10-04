"""One-time F1TV sign-in so the 'F1 live timing' source can connect.

    python -m src.pitwall.f1tv_login          # sign in (opens a login URL)
    python -m src.pitwall.f1tv_login status   # show token status
    python -m src.pitwall.f1tv_login logout   # remove the stored token

Uses FastF1's own flow: you sign in to your F1TV account in your browser and
the subscription token is stored locally by FastF1 (never by PitWall). An
active F1TV Access/Pro/Premium subscription is required by F1 for live timing.
"""
from __future__ import annotations

import sys

from fastf1.internals import f1auth


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "login"
    if cmd == "status":
        f1auth.print_auth_status()
    elif cmd == "logout":
        f1auth.clear_auth_token()
        print("F1TV token removed.")
    else:
        token = f1auth.get_auth_token()
        print("Signed in — the live-timing source is ready." if token else "Sign-in did not complete.")


if __name__ == "__main__":
    main()
