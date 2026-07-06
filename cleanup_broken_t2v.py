"""
cleanup_broken_t2v.py

Surgically removes all artifacts from the broken T→V run (single pooled vision key,
seeds 456/789/1337) so the pipeline treats it as never-run.

What this removes:
  1. checkpoint directories: checkpoints/Cross-Attention_T2V_seed_{456,789,1337}/
  2. completed_run keys:    "456:Cross-Attention T→V", "789:Cross-Attention T→V",
                            "1337:Cross-Attention T→V"
  3. results entry:         results["Cross-Attention T→V"]   (all metric arrays)

Dry-run by default — pass --execute to actually delete.
"""

import json, os, sys, shutil, argparse

PROGRESS_PATH  = "./results/experiment_progress.json"
CKPT_DIR       = "./checkpoints"
ARCH_NAME      = "Cross-Attention T\u2192V"          # exact key used in JSON
BROKEN_SEEDS   = [456, 789, 1337]
BROKEN_RUN_KEYS = {f"{s}:{ARCH_NAME}" for s in BROKEN_SEEDS}
BROKEN_CKPT_DIRS = [
    os.path.join(CKPT_DIR, f"Cross-Attention_T2V_seed_{s}")
    for s in BROKEN_SEEDS
]

def main(execute: bool):
    mode = "EXECUTE" if execute else "DRY-RUN"
    print(f"\n{'='*60}")
    print(f"  cleanup_broken_t2v.py  [{mode}]")
    print(f"{'='*60}\n")

    # ── 1. Checkpoint directories ──────────────────────────────────────────────
    print("── Step 1: Checkpoint directories ──────────────────────────")
    for d in BROKEN_CKPT_DIRS:
        if os.path.exists(d):
            files = []
            for root, _, fnames in os.walk(d):
                for f in fnames:
                    files.append(os.path.join(root, f))
            size_mb = sum(os.path.getsize(fp) for fp in files) / 1e6
            print(f"  FOUND  {d}  ({size_mb:.0f} MB, {len(files)} file(s))")
            if execute:
                shutil.rmtree(d)
                print(f"  DELETED {d}")
        else:
            print(f"  ABSENT {d}  (nothing to do)")

    # ── 2. experiment_progress.json ───────────────────────────────────────────
    print("\n── Step 2: experiment_progress.json ─────────────────────────")
    if not os.path.exists(PROGRESS_PATH):
        print(f"  {PROGRESS_PATH} not found — nothing to clean.")
        return

    with open(PROGRESS_PATH, "r") as f:
        data = json.load(f)

    original_completed = set(data.get("completed_runs", []))
    original_results   = set(data.get("results", {}).keys())

    to_remove_runs   = original_completed & BROKEN_RUN_KEYS
    has_results_key  = ARCH_NAME in data.get("results", {})

    print(f"  completed_runs to remove ({len(to_remove_runs)}): {sorted(to_remove_runs)}")
    print(f"  results['{ARCH_NAME}'] present: {has_results_key}")

    if execute:
        # Remove broken run keys
        data["completed_runs"] = sorted(original_completed - BROKEN_RUN_KEYS)
        # Remove results entry entirely
        if ARCH_NAME in data.get("results", {}):
            del data["results"][ARCH_NAME]

        # Write back atomically (write temp then rename)
        tmp = PROGRESS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, PROGRESS_PATH)
        print(f"\n  Written cleaned JSON to {PROGRESS_PATH}")
    else:
        print(f"\n  [DRY-RUN] No changes written.")

    # ── 3. Verification read-back ─────────────────────────────────────────────
    if execute:
        print("\n── Step 3: Read-back verification ───────────────────────────")
        with open(PROGRESS_PATH, "r") as f:
            verify = json.load(f)

        remaining_t2v_runs = [r for r in verify["completed_runs"] if "T\u2192V" in r]
        t2v_in_results     = "Cross-Attention T\u2192V" in verify.get("results", {})
        broken_ckpts_exist = [d for d in BROKEN_CKPT_DIRS if os.path.exists(d)]

        print(f"  Remaining T\u2192V entries in completed_runs: {remaining_t2v_runs}")
        print(f"  'Cross-Attention T\u2192V' in results:         {t2v_in_results}")
        print(f"  Broken checkpoint dirs still on disk:      {broken_ckpts_exist}")

        if not remaining_t2v_runs and not t2v_in_results and not broken_ckpts_exist:
            print("\n  \u2705 All broken T\u2192V artifacts cleanly removed.")
            print("  Pipeline will treat this as a fresh, never-run architecture.")
        else:
            print("\n  \u274c WARNING: some artifacts may remain — review above.")

    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="Actually delete files and update JSON (default: dry-run)")
    args = parser.parse_args()
    main(execute=args.execute)
