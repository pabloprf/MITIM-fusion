'''
Per-iteration namelist overrides (`extraOptions_special`, `allocation_special`): dicts keyed by
an iteration selector, merged on top of the baseline value for the iteration being evaluated.
Selector syntax and examples live in templates/namelist.portals.yaml.
'''

from mitim_tools.misc_tools.LOGtools import printMsg as print


class IterationSelector:
    '''A spec key: "N" (exact), ">N", ">=N", "<N", "<=N".'''

    COMPARISONS = (
        (">=", lambda a, b: a >= b),
        ("<=", lambda a, b: a <= b),
        (">",  lambda a, b: a >  b),
        ("<",  lambda a, b: a <  b),
    )

    @classmethod
    def is_range(cls, key):
        return any(str(key).strip().startswith(op) for op, _ in cls.COMPARISONS)

    @classmethod
    def matches(cls, key, iteration, label="CGYRO *_special"):
        '''Malformed keys never match and print a warning.'''
        key = str(key).strip()
        # PORTALS sources the evaluation number from the Dakota-style filename: a string in the
        # Execution phase, an int during the SR initializer.
        try:
            iteration = int(iteration)
        except (TypeError, ValueError):
            print(f"\t- [{label}] Non-integer evaluation number {iteration!r}; no per-iteration override applied", typeMsg='w')
            return False

        for op, cmp in cls.COMPARISONS:
            if key.startswith(op):
                try:
                    return cmp(iteration, int(key[len(op):].strip()))
                except ValueError:
                    print(f"\t- [{label}] Malformed spec key {key!r}; ignoring", typeMsg='w')
                    return False
        try:
            return iteration == int(key)
        except ValueError:
            print(f"\t- [{label}] Malformed spec key {key!r}; expected integer or comparison (e.g. '5', '>5'); ignoring", typeMsg='w')
            return False


class PerIterOverrides:
    '''
    One `*_special` spec. Range keys are applied first and exact-integer keys last, so "5" wins
    over ">4" for iteration 5.
    '''

    def __init__(self, spec, legacy_spec=None, label="extraOptions_special"):
        self.label = f"CGYRO {label}"
        if not spec and legacy_spec:
            legacy_name = label.replace("_special", "_first")
            print(
                f"\t- [{self.label}] `{legacy_name}` is deprecated; treating as "
                f"`{label}: {{\"0\": ...}}`. Please update your namelist.",
                typeMsg='w',
            )
            spec = {"0": legacy_spec}
        self.spec = spec or {}

    @classmethod
    def from_run_options(cls, run_options, name):
        '''`<name>_special`, falling back to the retired `<name>_first` alias.'''
        run_options = run_options or {}
        return cls(run_options.get(f"{name}_special"), run_options.get(f"{name}_first"), label=f"{name}_special")

    def merge(self, baseline, iteration):
        '''
        The overrides matching `iteration` on top of `baseline`, as a NEW dict: the shared
        namelist dict is never mutated and never handed back by reference (PORTALS re-reads
        the namelist every iteration, so a leak would carry overrides forward).

        A None baseline with no match stays None, so the callee still applies its own default
        (SIMtools.run only sizes the allocation itself when `allocation is None`).
        '''
        matched, merged = [], dict(baseline) if baseline else {}

        for wanted_range in (True, False):
            for key in self.spec:
                if IterationSelector.is_range(key) is not wanted_range:
                    continue
                if not IterationSelector.matches(key, iteration, label=self.label):
                    continue
                matched.append(str(key))
                merged.update(self.spec[key] or {})

        if not matched:
            return None if baseline is None else dict(baseline)

        self._announce(matched, baseline, merged, iteration)
        return merged

    def _announce(self, matched, baseline, merged, iteration):
        baseline = baseline or {}
        added = {k: merged[k] for k in sorted(set(merged) - set(baseline))}
        changed = {k: merged[k] for k in sorted(k for k in merged if k in baseline and baseline[k] != merged[k])}
        print(
            f"\n- [{self.label}] Iteration {iteration}: overrides from keys {matched} "
            f"-> added {added}, changed {changed}",
            typeMsg='i',
        )
