"""Mechanics the CLI Design Standard requires, independent of any command or the domain library.

Nothing in this package imports from `provibench.commands`, `provibench.bench`, or
`provibench.app`; a test enforces that rule so the package stays reusable as the base of
other tools.
"""
