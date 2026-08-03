"""``python -m hamqadam_ai.workers`` - run one verification worker.

The container's worker command. Kept separate from ``consumer.py`` so importing
the consumer never starts one.
"""

from __future__ import annotations

from hamqadam_ai.workers.consumer import main

if __name__ == "__main__":
    main()
