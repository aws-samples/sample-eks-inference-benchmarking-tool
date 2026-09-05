#!/usr/bin/env python3
"""
Command-line interface for the EKS Inference Benchmarking Tool.

ORGANIZATION:
    1. Help styling - a Flags heading and tree-ordered command listing
    2. Root group - the top-level command
    3. Command builder - construct one Click command per tree entry
    4. Registration - attach every command to the root group

ARCHITECTURE:
    Click owns parsing, dispatch and help for the command tree. The vocabulary itself lives in
    utils.COMMAND_TREE, and each command execs the unchanged `app/bench/src/main.py`, so the
    argparse layer and the in-cluster invocation are untouched.

    Per-command flags are not declared here yet: they are still owned by each legacy command's
    argparse parser, and `--help` on a command delegates to it. Declaring them as Click options
    is the next step and will remove that delegation.
"""

import os
import sys
from importlib.metadata import PackageNotFoundError, version

import click
from rich.console import Console

# Allow running via `python3 app/cli.py ...` without installing the package.
# When installed, the console script already has the correct import path.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.utils import (
    COMMAND_TREE,
    DISTRIBUTION_NAME,
    HELP_FLAGS,
    PROG_NAME,
    Leaf,
    execute_leaf,
    print_leaf_help,
)

console = Console()
stderr_console = Console(stderr=True)


def _version() -> str:
    """
    Resolve the installed version, falling back when running from an uninstalled clone.

    Returns:
        The distribution version, or "0.0.0+local" when the package is not installed
    """
    try:
        return version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return "0.0.0+local"


# =============================================================================
# Help Styling
# =============================================================================


class FlagsHelpFormatter(click.HelpFormatter):
    """Click help formatter that labels the option section "Flags"."""

    def write_heading(self, heading: str) -> None:
        """
        Write a section heading, renaming Click's "Options" to "Flags".

        Args:
            heading: The section heading Click is about to write
        """
        super().write_heading("Flags" if heading == "Options" else heading)


class CliContext(click.Context):
    """Click context that renders help with FlagsHelpFormatter.

    Click 8 builds the formatter from the context rather than the command, so this is the hook
    that takes effect. Overriding make_formatter on a Command or Group has no effect, because
    Command.get_help calls ctx.make_formatter().
    """

    formatter_class = FlagsHelpFormatter


class CliGroup(click.Group):
    """Click group whose contexts use CliContext, listing commands in tree order."""

    context_class = CliContext

    def list_commands(self, ctx: click.Context) -> list[str]:
        """
        List commands in the order they were registered rather than alphabetically.

        The tree is ordered deliberately, with the most-used command first, and Click sorts
        by default.

        Args:
            ctx: Click context, unused but required by the base signature

        Returns:
            Command names in registration order
        """
        return list(self.commands)


class CliCommand(click.Command):
    """Click command whose contexts use CliContext."""

    context_class = CliContext


# =============================================================================
# Root Group
# =============================================================================


@click.group(
    cls=CliGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
    epilog=f'Use "{PROG_NAME} [command] --help" for more information about a command.',
)
# prog_name is left unset so --version reports whichever console script alias was invoked.
@click.version_option(version=_version())
@click.pass_context
def main(ctx: click.Context) -> None:
    """
    Benchmark and validate self-managed vLLM inference with reproducible protocols.

    Most commands come in an in-cluster form and a direct form. The in-cluster form runs the
    client as a Kubernetes Job beside the server and reaches it over cluster DNS, which is what
    every runbook uses and what any time-to-first-token figure requires. The direct form runs
    the client wherever you type it against an endpoint you supply, for vLLM on a bare EC2
    instance or a Service you can already reach.

    \f

    Args:
        ctx: Click context, used to detect whether a subcommand was given
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# =============================================================================
# Command Builders
# =============================================================================


def build_command(path: str, leaf: Leaf) -> click.Command:
    """
    Build one runnable command that either shows help or execs the harness.

    Unknown options pass through untouched, because the legacy argparse parser owns flag
    validation. A help flag is intercepted here instead, so help renders without running.

    Args:
        path: The command name as typed, e.g. "capture-reference"
        leaf: The tree entry this command exposes

    Returns:
        A Click command ready to attach to the root group or a group
    """

    @click.command(
        name=leaf.name,
        cls=CliCommand,
        help=leaf.summary,
        add_help_option=False,
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    )
    @click.argument("arguments", nargs=-1, type=click.UNPROCESSED)
    def command(arguments: tuple[str, ...]) -> None:
        """
        Show this command's help, or exec its legacy command.

        Args:
            arguments: Everything after the command path, forwarded verbatim
        """
        try:
            if any(argument in HELP_FLAGS for argument in arguments):
                print_leaf_help(path, leaf)
                return
            execute_leaf(leaf, arguments)
        except click.ClickException:
            raise
        except Exception as error:
            stderr_console.print(f"[red]Error:[/red] {error}")
            sys.exit(1)

    return command


# =============================================================================
# Registration
# =============================================================================

for _name, _leaf in COMMAND_TREE.items():
    main.add_command(build_command(_name, _leaf))


if __name__ == "__main__":
    # Ensure help/usage matches the installed console script name.
    main(prog_name=PROG_NAME)
