# Contributing guidelines

## Workflow

Every change to the repository should come out of an issue where the change is
discussed.

[Pull requests](#pull-requests) should always follow from the discussion in an
existing issue. The only exception are minor, obvious changes such as fixing
typos in the documentation.

The discussion in a pull request should only be about the low-level details of
its implementation. All high-level, conceptual discussion belongs in the issue
that is addressed by the pull request.

## Pull requests

Pull requests to the `main` branch should have titles of the following form:

    gh-NN: Subject line

The prefix `gh-NN` should refer to the issue number (`NN`) that is addressed
by the pull request.

The body of the pull request should contain a description of the changes, and
any relevant details or caveats of the implementation.

The pull request should not repeat or summarise the discussion of its
associated issue. Instead, it should link to the issue using git's so-called
"trailers". These are lines of the form `key: value` which are at the end of
the pull request description, separated from the message body by a blank line.

To generically refer to an issue without any further action, use `Refs` and
one or more GitHub issue numbers:

    Refs: #12
    Refs: #25, #65

To indicate that the pull request shall close an open issue, use `Closes` and
a single GitHub issue number:

    Closes: #17

You can use any of the other common git trailers. In particular, you can use
`Cc` to notify others of your pull request via their GitHub usernames:

    Cc: @octocat

## Versioning

The current version number is automatically inferred from the last release,
subsequent unreleased commits, and local changes, if any.

## Releasing

New versions of _Heracles_ are prepared using GitHub releases.

Creating a GitHub release will automatically start the build process that
uploads Python packages for the new version to `PyPI`.
