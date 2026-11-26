from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vietmailguard.v21.multilingual_dl_experiment import offline_boundary
from vietmailguard.v23.train_guard import audit_expansion

if __name__ == "__main__":
    with offline_boundary(ROOT):
        audit_expansion(ROOT)
