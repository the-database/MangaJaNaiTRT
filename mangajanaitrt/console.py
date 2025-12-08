from rich.console import Console

console = Console()
DEBUG = False


def dbg(msg: str) -> None:
    if DEBUG:
        console.log(f"[dim]{msg}[/]")
