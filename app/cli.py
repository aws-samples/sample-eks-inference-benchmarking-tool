#!/usr/bin/env python3
"""Console-pretty front end for the serving-sweep benchmark tool."""
from __future__ import annotations

import os
import subprocess
import sys

from utils import (
    CLI_USAGE,
    HARNESS_COMMANDS,
    HARNESS_FLAGS,
    HELP_FLAGS,
    MAIN_SCRIPT,
    PARSER_COMMANDS,
    benchmark_main,
)

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text


def _console() -> Console:
    """Build a console that respects standard no-color behavior."""
    return Console(no_color=bool(os.environ.get("NO_COLOR")), force_terminal=None)


def _print_root_help() -> None:
    """Render the complete command inventory."""
    console = _console()
    console.print("[bold]serving-sweep[/bold]")
    console.print("Benchmark and validate self-managed vLLM inference with reproducible protocols.")
    console.print()
    console.print("[bold]Usage:[/bold]")
    console.print(f"  {CLI_USAGE}", markup=False)
    console.print()
    console.print("[bold]Commands:[/bold]")
    table = Table(show_header=False, box=None, padding=(0, 2), collapse_padding=True)
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()
    for name, description in benchmark_main.COMMANDS.items():
        table.add_row(name, description)
    console.print(table)
    console.print()
    console.print("Run [cyan]python3 app/cli.py <command> --help[/cyan] for command flags.")


def _print_simple_help(command: str) -> None:
    """Render help for commands that do not own an argparse parser."""
    console = _console()
    console.print(f"[bold]Usage:[/bold] python3 app/cli.py {command} [flags]")
    console.print()
    console.print(benchmark_main.COMMANDS[command])
    console.print()
    console.print("[bold]Flags:[/bold]")
    table = Table(show_header=False, box=None, padding=(0, 2), collapse_padding=True)
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()
    if command in HARNESS_COMMANDS:
        for names, description in HARNESS_FLAGS:
            table.add_row(names, description)
    table.add_row("-h, --help", "Show this message and exit")
    console.print(table)


def _print_styled_parser_output(command: str, output: str) -> None:
    """Style captured argparse help without reconstructing its option definitions."""
    console = _console()
    explicit = f"usage: main.py {command}"
    replacement = f"usage: python3 app/cli.py {command}"
    output = output.replace(explicit, replacement, 1) if explicit in output else output.replace(
        "usage: main.py", replacement, 1
    )
    for line in output.rstrip().splitlines():
        stripped = line.strip().lower()
        if stripped in ("options:", "optional arguments:"):
            console.print("Flags:", style="bold")
            continue
        if stripped == "positional arguments:":
            console.print("Positional Arguments:", style="bold")
            continue
        if stripped.startswith("usage:"):
            console.print(line, style="bold")
            continue
        text = Text(line)
        text.highlight_regex(r"(?<!\w)--?[a-zA-Z][a-zA-Z0-9-]*", style="cyan")
        console.print(text, soft_wrap=True)


def _print_parser_help(command: str) -> None:
    """Capture one existing parser's help and render it without running the command."""
    completed = subprocess.run(
        [sys.executable, str(MAIN_SCRIPT), command, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise click.ClickException(f"Unable to render help for {command!r}: {detail}")
    _print_styled_parser_output(command, completed.stdout)


def _print_command_help(command: str) -> None:
    """Render help from the existing parser or simple command metadata."""
    if command in PARSER_COMMANDS:
        _print_parser_help(command)
    else:
        _print_simple_help(command)


def _execute(command: str, arguments: tuple[str, ...]) -> None:
    """Replace this process with the unchanged benchmark entry point."""
    argv = [sys.executable, str(MAIN_SCRIPT), command, *arguments]
    os.execv(sys.executable, argv)


@click.group(invoke_without_command=True, add_help_option=False)
@click.option("-h", "--help", "show_help", is_flag=True, is_eager=True)
@click.pass_context
def cli(ctx: click.Context, show_help: bool) -> None:
    """Run or inspect serving-sweep benchmark commands."""
    if show_help or ctx.invoked_subcommand is None:
        _print_root_help()


def _command_for(name: str) -> click.Command:
    """Build one dynamic command that either shows help or execs main.py."""

    @click.command(
        name=name,
        add_help_option=False,
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    )
    @click.argument("arguments", nargs=-1, type=click.UNPROCESSED)
    def command(arguments: tuple[str, ...]) -> None:
        if any(argument in HELP_FLAGS for argument in arguments):
            _print_command_help(name)
            return
        _execute(name, arguments)

    command.help = benchmark_main.COMMANDS[name]
    return command


for command_name in benchmark_main.COMMANDS:
    cli.add_command(_command_for(command_name))


if __name__ == "__main__":
    cli(prog_name="serving-sweep")
