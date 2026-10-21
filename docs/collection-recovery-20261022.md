# Historical collection recovery — 2026-10-22 JST

## Evidence and diagnosis

- The old checkpoint recorded 112 attempted and failed days, zero archived days, yet advanced to 2023-04-23.
- Run 39635727679, job 119349750121, processed 2023-02-26 through 2023-03-04 with `discovered_races: 0`, no usable rows and no recorded HTTP error. That run did not retain HTTP response evidence, so its original response-level cause cannot be established retrospectively.
- The day collector formed `kaisai_group_id` from the requested date and venue. This is incorrect for later days: the ID identifies the opening day of a meeting.
- The early diagnostic in commit 9ab491e7 had an overescaped regex. Its log reported zero regex matches while the same HTML contained 48 real race links. This diagnostic issue is separate from the old collector's regex; do not claim it caused every empty day.
- Failed days advanced the historical cursor even when every day in a batch had zero races. Empty parsed programs and unsuccessful HTTP bodies were also previously cacheable; recent upstream fixes already addressed those two cache issues.
- The source entry table now has an `H` column. Positional parsing shifted win rate, place rate, gear and comment. Named-header parsing fixes this.

## Live proof

On 2026-10-22, both public pages returned HTTP 200:

- `/db/race_program/?kaisai_group_id=2023090573`: 36 unique published IDs.
- `/race/special/race_info/?kaisai_group_id=2023090573`: the same 36 IDs.
- Each of 2023-09-05, 06 and 07 contains 12 IDs. The browser also displayed all three date tabs.
- Each day's first race was parsed live: 7, 6 and 7 entries respectively, with the same result-row counts and real payouts. For 202309057301 the trifecta was 1>7>2, 44,160 yen; this was checked against the browser's result table.
- A fresh calendar/program discovery for 2023-01-01 returned 96 IDs across nine venues, with no discovery error. December 2022 meeting openings are covered by the previous-month calendar.

## Implementation and operation

- Parse IDs from actual anchor query parameters, not arbitrary strings or generated race-number ranges.
- Use monthly public calendars to identify active venues and their opening dates, then filter published meeting IDs to the requested date.
- Use versioned parsed caches; never trust an empty list as durable negative discovery.
- Keep original parsed entry/result/payout objects in `source_records` inside encrypted SQLite, in addition to the existing upstream model tables.
- A per-race exception rolls back that race and continues the day. Reports count committed database rows.
- Partial days are archived with a partial status; they are not reported as complete model-training dates.
- Empty days stop their shard. An all-empty batch cannot advance the fast cursor or automatically continue, and the workflow fails visibly.
- Only nonzero days with a successful permanent monthly Git-branch upload count as archived during checkpoint finalization.
- The existing private `COLLECTOR_PASSPHRASE` protects AES256 GPG files; no plaintext database or HTML cache is uploaded.
- The historical cursor was reset to 2023-01-01 while retaining all 112 failed dates for audit. Successful complete recollections remove dates from that failure list.

`recover-historical-collection.yml` first verifies the requested meeting, collects all races on 2023-09-06, requires a complete nonzero report, encrypts and publishes the day, then dispatches the eight-shard historical workflow. A failed pilot cannot restart bulk collection.

Regression checks: link query parsing and HTML entities, second-day opening-ID selection, disjoint shards, durable partial-day accounting and all-empty cursor preservation. Five checks passed locally and on the recovery runner.

## Recovery commits

- 7e3ca44d7477cab19627e43a167a5441a2ae0e00: discovery, parser, checkpoint and live recovery workflow.
- b1cc0c5c0df2d5bddb5e85e36313cc2807ee7fbf: visible empty-batch failures and per-ten-race progress logs.

## One-day Actions result and archival correction

Recovery run 39693046127 collected all 78 published races on 2023-09-06: 538 entries, 532 classified result rows and 78 payout records; collection errors were zero. The six-row difference between entries and classified finishes is retained in the audit rather than equating starters with numeric finish positions.

Release creation returned `HTTP 403: Resource not accessible by integration` despite declared Contents write permission. The encrypted pilot was preserved in artifact 12390246700. The permanent publisher therefore uses existing Contents permission to write encrypted files and numeric audits to dedicated `collected-data-YYYY-MM` Git branches. Non-forced ref updates and retries preserve simultaneous shards; no repository permission is expanded. The pilot is restored from the same recovery run, decrypted only on the runner for a SQLite integrity/count check, then permanently archived. It is freshly collected data from this task, not an imported legacy dataset.
