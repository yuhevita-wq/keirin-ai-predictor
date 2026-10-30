"""V2 retired by user instruction on 2026-10-30. Textbook policy is current."""
from ura.manual_formations import main, POLICY_ID

def decide(*args, **kwargs):
    raise RuntimeError("V2 six-ticket rule retired. Use textbook manual formations with full independent score/probability/odds snapshots")

if __name__ == '__main__':
    main()
