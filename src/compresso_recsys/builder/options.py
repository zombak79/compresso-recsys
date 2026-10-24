"""Adapter-specific dataset options: validation, coercion and the subset notice.

An adapter's constructor arguments that the builder does not already own can be
set per build, from Python as a mapping or from the command line as repeated
``--dataset_option key=value`` flags. The adapter's own signature and
annotations are the source of truth, so a new option needs no registration here.
"""

from __future__ import annotations

import collections.abc
import inspect
import types
import typing
from typing import Any


#: Adapter constructor arguments the builder already owns. They are configured
#: through the builder's dedicated flags, so exposing them again as dataset options
#: would let one build be told two different things.
_BUILDER_MANAGED_DATASET_OPTIONS = frozenset({
    "self", "data_dir", "metadata_text_fields", "min_entity_text_words",
    "show_progress", "category", "include_image_urls",
})


def _dataset_option_names(cls) -> list[str]:
    """Adapter constructor arguments a caller may set for ``cls``."""
    try:
        parameters = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins have no signature
        return []
    return [
        name for name, parameter in parameters.items()
        if name not in _BUILDER_MANAGED_DATASET_OPTIONS
        and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    ]


def _coerce_dataset_option(value, annotation):
    """Parse one command-line option value using the adapter's own annotation.

    Only strings are converted, so a Python caller that already passes a float
    or a tuple keeps exactly what it passed. The annotation is the adapter's, so
    a new option becomes settable from the command line by being annotated
    rather than by being registered anywhere here.
    """
    if not isinstance(value, str) or annotation is inspect.Parameter.empty:
        return value
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        members = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        # ``key=`` clears an optional option. Only the empty string means this,
        # so the literal word "none" is still usable as a value.
        if len(members) < len(typing.get_args(annotation)) and not value.strip():
            return None
        return _coerce_dataset_option(value, members[0]) if members else value
    if (
        origin is not None
        and isinstance(origin, type)
        and not issubclass(origin, str)
        and issubclass(origin, collections.abc.Sequence)
    ):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if annotation is bool:
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"expected true or false, got {value!r}")
    if annotation is int:
        return int(value)
    if annotation is float:
        return float(value)
    return value


#: What to pass to undo each registered option and read the whole source. The
#: builder cannot infer this: "all of it" is ``1.0`` for a fraction but an
#: absent bound for a window, and only the adapter knows which it has.
_FULL_SOURCE_OPTION = {
    "session_sample": "1.0",
    "user_sample": "1.0",
    "start": "",
    "end": "",
}


def _registered_subset_notice(spec, dataset: str, caller_options: dict[str, Any]) -> str | None:
    """Say so when a build silently reads a registered subset of its source.

    Only the registered defaults the caller did not set are reported: an option
    the caller set themselves is a deliberate choice and needs no warning, even
    when its value equals the registered one. Without this a bare build of OTTO
    looks exactly like a full one and quietly measures a tenth of the log.
    """
    registered = getattr(spec, "dataset_options", None) or {}
    in_force = {key: value for key, value in registered.items() if key not in caller_options}
    if not in_force:
        return None
    settings = ", ".join(f"{key}={value!r}" for key, value in sorted(in_force.items()))
    flags = " ".join(f"--dataset_option {key}={_FULL_SOURCE_OPTION.get(key, '')}"
                     for key in sorted(in_force))
    return (f"{dataset} is reading a registered subset of its source, not the whole of it: "
            f"{settings}. This is what the published default tables measure. "
            f"For everything, pass {flags} -- which needs substantially more memory.")


def _dataset_options_for(spec, args) -> dict[str, Any]:
    """The options to construct this dataset with: spec defaults, then the caller's.

    Resolving both through the same validator means a typo in the registry is
    caught exactly like a typo on the command line, and re-running it on an
    already-resolved mapping is a no-op, so it does not matter whether the
    caller reached _make_dataset through _resolve_args or built a namespace.
    """
    return {
        **_resolve_dataset_options(spec, getattr(spec, "dataset_options", None)),
        **_resolve_dataset_options(spec, getattr(args, "dataset_options", None)),
    }


def _resolve_dataset_options(spec, raw) -> dict[str, Any]:
    """Validate adapter options and coerce the ones that arrived as strings.

    Accepts either the mapping the Python API takes or the repeated
    ``--dataset_option key=value`` strings argparse collects. An unknown name is
    an error rather than a silently ignored keyword, because the failure it
    replaces -- a build that quietly used the default variant -- is invisible in
    the resulting checkpoint.
    """
    if not raw:
        return {}
    if isinstance(raw, collections.abc.Mapping):
        items = list(raw.items())
    else:
        items = []
        for entry in raw:
            key, separator, value = str(entry).partition("=")
            if not separator or not key.strip():
                raise ValueError(
                    f"dataset options must be given as key=value, got {entry!r}"
                )
            items.append((key.strip(), value))
    allowed = _dataset_option_names(spec.cls)
    try:
        hints = typing.get_type_hints(spec.cls.__init__)
    except Exception:  # pragma: no cover - unresolvable annotations
        hints = {}
    resolved: dict[str, Any] = {}
    for key, value in items:
        if key not in allowed:
            choices = ", ".join(allowed) if allowed else "no options"
            raise ValueError(
                f"unknown dataset option {key!r} for dataset {spec.cls.__name__}; "
                f"accepts: {choices}"
            )
        try:
            resolved[key] = _coerce_dataset_option(value, hints.get(key, inspect.Parameter.empty))
        except ValueError as error:
            raise ValueError(f"invalid value for dataset option {key!r}: {error}") from error
    return resolved
