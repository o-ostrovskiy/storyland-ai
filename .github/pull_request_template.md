## Why

<!-- The problem or review finding this addresses. Link the MYS ticket. -->

## What

<!-- The change, including any invariants it creates or relies on. -->

## Docs

<!-- What a reader will BELIEVE this covers, and where that's written down.
     Three obligations, not one:
     1. Update the doc where the claim lives (README runbook, workflow
        header, compose comment) in the SAME diff.
     2. State the control's EDGES at the inference site — an honest control
        still misleads if nobody writes what it does NOT cover where people
        form the assumption (MYS-611: "boots the full stack" read as
        "migrations validated" while the gate ran SQLite).
     3. CROSS-REPO COUNTS: if the claim lives in another repo's doc (for
        this repo, often storyland-infrastructure's deploy/README.md or a
        workflow header), the Docs entry is a pointer to the change there —
        having no LOCAL file to edit is not "no doc impact" (MYS-615).
     "No doc impact" is a valid entry if argued, not asserted. -->

## Verification

<!-- What was checked before push (parsers, bash -n, simulations), and what
     live proof looks like after merge/deploy. Derived numbers show their
     working next to the knobs they derive from. -->
